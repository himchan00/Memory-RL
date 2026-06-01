import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policies.seq_models import SEQ_MODELS
from policies.seq_models.Rff_embedding import RFFEmbedding
from policies.models.conditioning import FiLMConditioningStack, HyperConditioningStack
from torchkit.networks import Mlp, ImageEncoder, IdentityModule


class RNN_head(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = config_seq.seq_model.hidden_size 

        self.obs_shortcut = config_seq.obs_shortcut
        self.full_transition = config_seq.full_transition
        self.project_output = config_seq.project_output
        self.conditioning = getattr(config_seq, "conditioning", "concat")

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}, project_output={self.project_output}, conditioning={self.conditioning}")
        ### Build Model
        self.use_image_encoder = config_seq.use_image_encoder
        self.is_oracle_markov = (
            config_seq.seq_model.name == "markov" and config_seq.seq_model.is_oracle
        )
        self.context_dim = config_seq.seq_model.context_dim
        if self.use_image_encoder:
            img_cfg = config_seq.image_encoder
            self.image_encoder = ImageEncoder(
                image_shape=tuple(img_cfg.image_shape),
                embedding_size=img_cfg.embedding_size,
                channels=list(img_cfg.channels),
                kernel_sizes=list(img_cfg.kernel_sizes),
                strides=list(img_cfg.strides),
                from_flattened=True,
                normalize_pixel=True,
            )
            self.image_flat_dim = int(np.prod(img_cfg.image_shape))
            encoded_obs_dim = img_cfg.embedding_size 
            # For oracle Markov, `_encode_obs` re-attaches the context tail after the CNN
            if self.is_oracle_markov:
                encoded_obs_dim += self.context_dim
        else:
            self.image_encoder = None
            self.image_flat_dim = None
            encoded_obs_dim = obs_dim
        ## 1. Observation embedder, Transition embedder
        if self.obs_shortcut:
            # Single Mlp over the full input. For oracle Markov, the context tail is already part of `encoded_obs_dim`
            self.observ_embedder = Mlp(
                input_size=encoded_obs_dim,
                output_size=self.hidden_dim,
                **config_seq.embedder.to_dict(),
            )
        else:
            self.observ_embedder = None

        transition_size = 2 * encoded_obs_dim + action_dim + 1 if self.full_transition else encoded_obs_dim + action_dim + 1
        if config_seq.seq_model.name == "markov":
            self.transition_embedder = IdentityModule() # dummy, not used
        elif config_seq.seq_model.name == "mate_rff":
            rff_cfg = config_seq.seq_model
            self.transition_embedder = RFFEmbedding(
                input_dim=transition_size,
                embedding_dim=self.hidden_dim,
                kernel=rff_cfg.kernel,
                normalize_inputs=config_seq.embedder.normalize_inputs,
            )
        else:
            self.transition_embedder = Mlp(
                input_size=transition_size,
                output_size=self.hidden_dim,  # transition_embedding size is set equal to the hidden_dim for residual connection in gpt
                **config_seq.embedder.to_dict()
            )


        ## 2. build Sequence model
        self.seq_model = SEQ_MODELS[config_seq.seq_model.name](
            input_size=self.hidden_dim, **config_seq.seq_model.to_dict()
        )

        ## 2b. build conditioning stack (only when obs_shortcut and not concat)
        if self.conditioning != "concat":
            assert self.obs_shortcut, \
                "conditioning ∈ {film, hypernet} requires obs_shortcut=True"
            assert config_seq.seq_model.name != "markov", \
                "no memory to condition on for markov; use conditioning='concat'"
            if self.conditioning == "film":
                cls = FiLMConditioningStack
            elif self.conditioning == "hypernet":
                cls = HyperConditioningStack
            else:
                raise ValueError(f"Invalid conditioning type: {self.conditioning}")
            cond_hidden_sizes = tuple(
                getattr(config_seq, "conditioning_hidden_sizes", ())
            )
            self.conditioner = cls(
                in_dim=self.hidden_dim,                         # obs_embedder output
                out_dim=self.hidden_dim,                        # joint embedding dim
                hidden_sizes=cond_hidden_sizes,                 # n-block depth knob
                cond_dim=config_seq.seq_model.hidden_size,      # dim of h_t
            )
        else:
            self.conditioner = None

        if torch.cuda.is_available() and config_seq.get("compile", False):
            self.seq_model = torch.compile(self.seq_model)
            self.observ_embedder = torch.compile(self.observ_embedder) if self.observ_embedder is not None else None
            self.transition_embedder = torch.compile(self.transition_embedder)
            if self.image_encoder is not None:
                self.image_encoder = torch.compile(self.image_encoder)
            if self.conditioner is not None:
                self.conditioner = torch.compile(self.conditioner)

        ## 3. Set embedding size
        if config_seq.seq_model.name == "markov":
            self.hidden_dim = 0  # no hidden state for Markov model
        self.embedding_size = self.hidden_dim
        if self.obs_shortcut:
            if self.conditioning == "concat":
                # cat(obs_emb, h_t); oracle-markov context tail is part of obs_emb now
                self.embedding_size += config_seq.seq_model.hidden_size
            else:
                # conditioner output replaces cat(obs_emb, h_t)
                self.embedding_size = self.conditioner.out_dim

    def _encode_obs(self, observs):
        """Run the image encoder on the image part of the observation.

        For oracle Markov runs with a CNN, the wrapper appends a `context_dim`
        tail to the flattened image. That tail must bypass the CNN and be
        re-attached so the single obs embedder receives the full input
        (image features + context).
        """
        if self.image_encoder is None:
            return observs
        if self.is_oracle_markov:
            image_part = observs[..., : self.image_flat_dim]
            context_part = observs[..., self.image_flat_dim :]
            encoded = self.image_encoder(image_part)
            return torch.cat([encoded, context_part], dim=-1)
        return self.image_encoder(observs)

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None, obs_embeds=None, transition_mask=None
    ):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        transition_mask: optional (T+1, B, 1) mask of valid transitions (used only for InputNorm stats)
        Outputs:
        hidden[t] = h_t: (T+1, B, dim)
        """
        observs_t = observs[:-1] # o[t]
        observs_t_1 = observs[1:] # o[t+1]
        if self.full_transition:
            inputs = self.transition_embedder(torch.cat((observs_t, actions, rewards, observs_t_1 - observs_t), dim=-1), mask=transition_mask)
        elif self.obs_shortcut:
            inputs = self.transition_embedder(torch.cat((observs_t, actions, rewards), dim=-1), mask=transition_mask)
        else:
            inputs = self.transition_embedder(torch.cat((actions, rewards, observs_t_1), dim=-1), mask=transition_mask)

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1], training = True
            )
            if self.obs_shortcut:
                inputs = inputs[1:] # skip the dummy transition at t = -1
                if self.seq_model.name in ("mate", "mate_rff"):
                    h0 = self.seq_model.internal_state_to_hidden(initial_internal_state) # (1, B, hidden_size)
                else:
                    h0 = inputs.new_zeros((1, inputs.shape[1], self.hidden_dim))
                ret = self.seq_model(inputs, initial_internal_state, obs_emb=obs_embeds[2:] if obs_embeds is not None else None)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
                output_target = info.pop("_output_target", None)    # FIX: pop from info, not duplicate output
                output = torch.cat((h0, output), dim = 0)
                if output_target is not None:
                    output_target = torch.cat((h0, output_target), dim = 0)
            else:
                ret = self.seq_model(inputs, initial_internal_state, obs_emb=obs_embeds[1:] if obs_embeds is not None else None)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
                output_target = info.pop("_output_target", None)
            return output, output_target, info
        else:  # useful for one-step rollout
            ret = self.seq_model(
                inputs, initial_internal_state, obs_emb=obs_embeds[-1:] if obs_embeds is not None else None
            )
            output = ret[0]
            current_internal_state = ret[1]
            return output, current_internal_state

    def forward(self, actions, rewards, observs, masks=None):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        masks[t] = mask_{t-1}, shape (T+1, B, 1) — optional; used only for InputNorm stats
        Outputs:
        embedding[t] = h_{t-1} or (h_{t-1}, o_{t-1}): (T+2, B, dim)
        """
        assert actions.dim() == rewards.dim() == observs.dim() == 3
        assert actions.shape[0] + 1 == rewards.shape[0] + 1  == observs.shape[0]
        # Build per-tensor InputNorm masks from the rollout mask if available.
        if masks is not None:
            transition_mask = masks  # aligns 1:1 with the (T+1, B, ·) transition tensor
            obs_mask = torch.cat((masks, masks[-1:]), dim=0)  # repeat last mask for the trailing obs at t=T
        else:
            transition_mask = None
            obs_mask = None

        observs = self._encode_obs(observs)
        if self.obs_shortcut:
            observs_embeds = self.observ_embedder(observs, mask=obs_mask)
        else:
            observs_embeds = None
        hidden_states, hidden_states_target, info = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs, obs_embeds=observs_embeds,
            transition_mask=transition_mask,
        )  # (T+1, B, dim)
        h_dummy = hidden_states.new_zeros((1, hidden_states.shape[1], hidden_states.shape[2]))
        hidden_states = torch.cat((h_dummy, hidden_states), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs

        if hidden_states_target is not None:
            hidden_states_target = torch.cat((h_dummy, hidden_states_target), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs

        if self.obs_shortcut and self.conditioning != "concat":
            # film/hypernet: conditioner first, then optional project_output on its output
            joint_embeds = self.conditioner(observs_embeds, hidden_states)
            joint_embeds_target = (
                self.conditioner(observs_embeds, hidden_states_target)
                if hidden_states_target is not None else None
            )
            if self.project_output:
                joint_embeds = F.normalize(joint_embeds, p=2, dim=-1) * np.sqrt(self.hidden_dim)
                if joint_embeds_target is not None:
                    joint_embeds_target = F.normalize(joint_embeds_target, p=2, dim=-1) * np.sqrt(self.hidden_dim)
        else:
            # concat (default) or no obs_shortcut: existing behavior — normalize each part, then cat
            if self.project_output:
                hidden_states = F.normalize(hidden_states, p=2, dim=-1) * np.sqrt(self.hidden_dim) # normalize the hidden states
                if hidden_states_target is not None:
                    hidden_states_target = F.normalize(hidden_states_target, p=2, dim=-1) * np.sqrt(self.hidden_dim)
                observs_embeds = F.normalize(observs_embeds, p=2, dim=-1) * np.sqrt(self.hidden_dim) if observs_embeds is not None else None
            if self.obs_shortcut:
                joint_embeds = torch.cat((observs_embeds, hidden_states), dim = -1) # Q(s, h)
                joint_embeds_target = torch.cat((observs_embeds, hidden_states_target), dim = -1) if hidden_states_target is not None else None
            else:
                joint_embeds = hidden_states # Q(h)
                joint_embeds_target = hidden_states_target

        if self.seq_model.hidden_size > 0: 
            d_forward = {}
            if not self.project_output:
                norms = hidden_states.detach().norm(dim=-1)
                d_forward["hidden_states_norm_mean"] = norms.mean(dim=1)
                d_forward["hidden_states_norm_std"] = norms.std(dim=1)
            d_forward.update(info)
        else: # To avoid warning when using Markov policy
            d_forward = {}
        if self.obs_shortcut:
            obs_norms = observs_embeds.detach().norm(dim=-1)
            d_forward["observs_embeds_norm_mean"] = obs_norms.mean(dim=1)
            d_forward["observs_embeds_norm_std"] = obs_norms.std(dim=1)

        return joint_embeds, joint_embeds_target, d_forward


    @torch.no_grad()
    def step(
        self,
        prev_internal_state,
        prev_action,
        prev_reward,
        prev_obs,
        obs,
        initial=False,
    ):
        """
        Used for evaluation (not training) so L=1
        prev_action a_{t-1}, (1, B, dim) 
        prev_reward r_{t-1}, (1, B, 1)
        prev_obs o_{t-1}, (1, B, dim)
        obs o_{t} (1, B, dim) 
        """
        assert prev_action.dim() == prev_reward.dim() == prev_obs.dim() == obs.dim() == 3
        bs = prev_action.shape[1]
        
        prev_obs = self._encode_obs(prev_obs)
        obs = self._encode_obs(obs)

        observs = torch.cat((prev_obs, obs), dim=0)
        if self.obs_shortcut:
            obs_embeds = self.observ_embedder(observs)
        else:
            obs_embeds = None

        if initial and self.obs_shortcut:
            current_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            if self.seq_model.name in ("mate", "mate_rff"):
                hidden_state = self.seq_model.internal_state_to_hidden(current_internal_state) # (1, B, hidden_size)
            else:
                hidden_state = prev_action.new_zeros((1, bs, self.hidden_dim))
        else:
            if initial:
                prev_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state, current_internal_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=observs,
                initial_internal_state=prev_internal_state,
                obs_embeds=obs_embeds,
            )
        hidden_state = hidden_state.squeeze(0)  # (B, dim)
        if self.obs_shortcut and self.conditioning != "concat":
            # film/hypernet: conditioner first, then optional project_output on its output
            joint_embed = self.conditioner(obs_embeds[-1], hidden_state)
            if self.project_output:
                joint_embed = F.normalize(joint_embed, p=2, dim=-1) * np.sqrt(self.hidden_dim)
        else:
            if self.project_output:
                hidden_state = F.normalize(hidden_state, p=2, dim=-1) * np.sqrt(self.hidden_dim) # normalize the hidden states
                obs_embeds = F.normalize(obs_embeds, p=2, dim=-1) * np.sqrt(self.hidden_dim) if obs_embeds is not None else None
            if self.obs_shortcut:
                joint_embed = torch.cat((obs_embeds[-1], hidden_state), dim = -1)
            else:
                joint_embed = hidden_state


        return joint_embed, current_internal_state
