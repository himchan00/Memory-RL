import torch
import torch.nn as nn
import numpy as np
from policies.seq_models import SEQ_MODELS
from policies.seq_models.Rff_embedding import RFFEmbedding
from policies.models.conditioning import (
    ConcatConditioner, FiLMConditioner, HyperConditioner,
)
from torchkit.networks import Mlp, ImageEncoder, IdentityModule, InputNorm


CONDITIONERS = {
    "concat": ConcatConditioner,
    "film": FiLMConditioner,
    "hypernet": HyperConditioner,
}


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
        self.conditioning = getattr(config_seq, "conditioning", "concat")
        assert self.conditioning in CONDITIONERS, f"Unknown conditioning {self.conditioning!r}"

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}, conditioning={self.conditioning}")
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

        ## 1. Externalized InputNorm (replaces the InputNorm that used to live inside Mlp / RFFEmbedding).
        self.encoded_obs_norm = InputNorm(encoded_obs_dim, skip=not config_seq.normalize_inputs) if self.obs_shortcut else None

        transition_size = 2 * encoded_obs_dim + action_dim + 1 if self.full_transition else encoded_obs_dim + action_dim + 1
        self.transition_input_norm = InputNorm(transition_size, skip=not config_seq.normalize_inputs)

        ## 2. Transition embedder
        if config_seq.seq_model.name == "markov":
            self.transition_embedder = IdentityModule() # dummy, not used
        elif config_seq.seq_model.name == "mate_rff":
            rff_cfg = config_seq.seq_model
            self.transition_embedder = RFFEmbedding(
                input_dim=transition_size,
                embedding_dim=self.hidden_dim,
                kernel=rff_cfg.kernel,
            )
        else:
            self.transition_embedder = Mlp(
                input_size=transition_size,
                output_size=self.hidden_dim,  # transition_embedding size is set equal to the hidden_dim for residual connection in gpt
                **config_seq.embedder.to_dict()
            )


        ## 3. build Sequence model
        self.seq_model = SEQ_MODELS[config_seq.seq_model.name](
            input_size=self.hidden_dim, **config_seq.seq_model.to_dict()
        )

        ## 4. build conditioning stack — unified for concat / film / hypernet.
        # cond_dim=0 for markov (no h_t); ConcatConditioner's cat reduces to plain MLP.
        self.cond_dim = 0 if config_seq.seq_model.name == "markov" else self.hidden_dim
        if self.obs_shortcut:
            self.conditioner = CONDITIONERS[self.conditioning](
                in_dim=encoded_obs_dim,
                out_dim=self.hidden_dim,
                hidden_sizes=(self.hidden_dim,) * getattr(config_seq, "conditioning_n_layer", 0),
                cond_dim=self.cond_dim,
            )
            self.embedding_size = self.conditioner.out_dim
        else:
            self.conditioner = None
            self.embedding_size = self.cond_dim

        if torch.cuda.is_available() and config_seq.get("compile", False):
            self.seq_model = torch.compile(self.seq_model)
            self.transition_embedder = torch.compile(self.transition_embedder)
            if self.image_encoder is not None:
                self.image_encoder = torch.compile(self.image_encoder)
            if self.conditioner is not None:
                self.conditioner = torch.compile(self.conditioner)

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
        self, actions, rewards, observs, initial_internal_state=None, transition_mask=None
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
            raw_transition = torch.cat((observs_t, actions, rewards, observs_t_1 - observs_t), dim=-1)
        elif self.obs_shortcut:
            raw_transition = torch.cat((observs_t, actions, rewards), dim=-1)
        else:
            raw_transition = torch.cat((actions, rewards, observs_t_1), dim=-1)

        if self.training:
            self.transition_input_norm.update_stats(raw_transition, mask=transition_mask)
        inputs = self.transition_embedder(self.transition_input_norm(raw_transition))

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1], training = True
            )
            if self.obs_shortcut:
                inputs = inputs[1:] # skip the dummy transition at t = -1
                if self.seq_model.name in ("mate", "mate_rff"):
                    h0 = self.seq_model.internal_state_to_hidden(initial_internal_state) # (1, B, hidden_size)
                else:
                    h0 = inputs.new_zeros((1, inputs.shape[1], self.cond_dim))  # 0-dim for markov
                ret = self.seq_model(inputs, initial_internal_state)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
                output_target = info.pop("_output_target", None)    # FIX: pop from info, not duplicate output
                output = torch.cat((h0, output), dim = 0)
                if output_target is not None:
                    output_target = torch.cat((h0, output_target), dim = 0)
            else:
                ret = self.seq_model(inputs, initial_internal_state)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
                output_target = info.pop("_output_target", None)
            return output, output_target, info
        else:  # useful for one-step rollout
            ret = self.seq_model(inputs, initial_internal_state)
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
            if self.training:
                self.encoded_obs_norm.update_stats(observs, mask=obs_mask)
            normalized_obs = self.encoded_obs_norm(observs)
        else:
            normalized_obs = None
        hidden_states, hidden_states_target, info = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs,
            transition_mask=transition_mask,
        )  # (T+1, B, dim)
        h_dummy = hidden_states.new_zeros((1, hidden_states.shape[1], hidden_states.shape[2]))
        hidden_states = torch.cat((h_dummy, hidden_states), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs

        if hidden_states_target is not None:
            hidden_states_target = torch.cat((h_dummy, hidden_states_target), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs

        if self.conditioner is not None:
            joint_embeds = self.conditioner(normalized_obs, hidden_states)
            joint_embeds_target = (
                self.conditioner(normalized_obs, hidden_states_target)
                if hidden_states_target is not None else None
            )
        else:
            joint_embeds = hidden_states # Q(h)
            joint_embeds_target = hidden_states_target

        if self.seq_model.hidden_size > 0 and hidden_states.shape[-1] > 0:
            d_forward = {}
            norms = hidden_states.detach().norm(dim=-1)
            d_forward["hidden_states_norm_mean"] = norms.mean(dim=1)
            d_forward["hidden_states_norm_std"] = norms.std(dim=1)
            d_forward.update(info)
        else: # To avoid warning when using Markov policy
            d_forward = {}

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
            if self.training:
                self.encoded_obs_norm.update_stats(observs)
            normalized_obs = self.encoded_obs_norm(observs)
        else:
            normalized_obs = None

        if initial and self.obs_shortcut:
            current_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            if self.seq_model.name in ("mate", "mate_rff"):
                hidden_state = self.seq_model.internal_state_to_hidden(current_internal_state) # (1, B, hidden_size)
            else:
                hidden_state = prev_action.new_zeros((1, bs, self.cond_dim))  # 0-dim for markov
        else:
            if initial:
                prev_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state, current_internal_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=observs,
                initial_internal_state=prev_internal_state,
            )
        hidden_state = hidden_state.squeeze(0)  # (B, dim)
        if self.conditioner is not None:
            joint_embed = self.conditioner(normalized_obs[-1], hidden_state)
        else:
            joint_embed = hidden_state


        return joint_embed, current_internal_state
