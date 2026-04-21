import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from policies.seq_models import SEQ_MODELS
from torchkit.networks import Mlp, ImageEncoder, gpt_like_Mlp


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

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}, project_output={self.project_output}")
        ### Build Model
        self.use_image_encoder = getattr(config_seq, 'image_encoder', None) is not None
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
            encoded_obs_dim = img_cfg.embedding_size
        else:
            self.image_encoder = None
            encoded_obs_dim = obs_dim
        ## 1. Observation embedder, Transition embedder
        if self.obs_shortcut:
            input_size = encoded_obs_dim
            self.observ_embedder = Mlp(input_size=input_size, output_size=self.hidden_dim,**config_seq.embedder.to_dict())
            self.observ_embedder = nn.Sequential(self.observ_embedder, gpt_like_Mlp(hidden_size=self.hidden_dim, n_layer=config_seq.seq_model.n_layer, pdrop=config_seq.seq_model.pdrop))
        else:
            self.observ_embedder = None

        transition_size = 2 * encoded_obs_dim + action_dim + 1 if self.full_transition else encoded_obs_dim + action_dim + 1
        if config_seq.seq_model.name == "markov":
            self.transition_embedder = nn.Identity() # dummy, not used
        else:
            self.transition_embedder = Mlp(
                input_size=transition_size,
                output_size=self.hidden_dim,  # transition_embedding size is set equal to the hidden_dim for residual connection in gpt
                **(config_seq.embedder.to_dict() | {'project_output': False}) # projection is not needed here
            )


        ## 2. build Sequence model
        self.seq_model = SEQ_MODELS[config_seq.seq_model.name](
            input_size=self.hidden_dim, **config_seq.seq_model.to_dict()
        )
        if torch.cuda.is_available():
            self.seq_model = torch.compile(self.seq_model)
            self.observ_embedder = torch.compile(self.observ_embedder) if self.observ_embedder is not None else None
            self.transition_embedder = torch.compile(self.transition_embedder)
            if self.image_encoder is not None:
                self.image_encoder = torch.compile(self.image_encoder)

        ## 3. Set embedding size
        if config_seq.seq_model.name == "markov":
            self.hidden_dim = 0  # no hidden state for Markov model
        self.embedding_size = self.hidden_dim
        if self.obs_shortcut:
            self.embedding_size += config_seq.seq_model.hidden_size # if obs_shortcut, the final embedding is the concatenation of obs embedding and hidden state
        

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None, obs_embeds=None
    ):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        Outputs:
        hidden[t] = h_t: (T+1, B, dim)
        """
        observs_t = observs[:-1] # o[t]
        observs_t_1 = observs[1:] # o[t+1]
        if self.full_transition:
            inputs = self.transition_embedder(torch.cat((observs_t, actions, rewards, observs_t_1), dim=-1))
        elif self.obs_shortcut:
            inputs = self.transition_embedder(torch.cat((observs_t, actions, rewards), dim=-1))
        else: 
            inputs = self.transition_embedder(torch.cat((actions, rewards, observs_t_1), dim=-1))

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1], training = True
            )
            if self.obs_shortcut:
                inputs = inputs[1:] # skip the dummy transition at t = -1
                h0 = inputs.new_zeros((1, inputs.shape[1], self.hidden_dim))
                ret = self.seq_model(inputs, initial_internal_state, obs_emb=obs_embeds[2:] if obs_embeds is not None else None)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
                output = torch.cat((h0, output), dim = 0) # add zero hidden state at t = 0
            else:
                ret = self.seq_model(inputs, initial_internal_state, obs_emb=obs_embeds[1:] if obs_embeds is not None else None)
                output = ret[0]
                info = ret[2] if len(ret) == 3 else {}
            return output, info
        else:  # useful for one-step rollout
            ret = self.seq_model(
                inputs, initial_internal_state, obs_emb=obs_embeds[-1:] if obs_embeds is not None else None
            )
            output = ret[0]
            current_internal_state = ret[1]
            return output, current_internal_state

    def forward(self, actions, rewards, observs):
        """
        Inputs: (Starting from dummy step at t = -1)
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        Outputs:
        embedding[t] = h_{t-1} or (h_{t-1}, o_{t-1}): (T+2, B, dim)
        """
        assert actions.dim() == rewards.dim() == observs.dim() == 3
        assert actions.shape[0] + 1 == rewards.shape[0] + 1  == observs.shape[0]
        
        if self.image_encoder is not None:
            observs = self.image_encoder(observs)
        if self.obs_shortcut:
            observs_embeds = self.observ_embedder(observs)
        else:
            observs_embeds = None
        hidden_states, info = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs, obs_embeds=observs_embeds
        )  # (T+1, B, dim)
        h_dummy = hidden_states.new_zeros((1, hidden_states.shape[1], hidden_states.shape[2]))
        hidden_states = torch.cat((h_dummy, hidden_states), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs
        if self.project_output:
            hidden_states = F.normalize(hidden_states, p=2, dim=-1) * np.sqrt(self.hidden_dim) # normalize the hidden states
        if self.obs_shortcut:
            joint_embeds = torch.cat((observs_embeds, hidden_states), dim = -1) # Q(s, h)
        else:
            joint_embeds = hidden_states # Q(h)

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

        return joint_embeds, d_forward


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
        
        if self.image_encoder is not None:
            prev_obs = self.image_encoder(prev_obs)
            obs = self.image_encoder(obs)

        observs = torch.cat((prev_obs, obs), dim=0)
        if self.obs_shortcut:
            obs_embeds = self.observ_embedder(observs)
        else:
            obs_embeds = None

        if initial and self.obs_shortcut:
            current_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
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
        if self.project_output:
            hidden_state = F.normalize(hidden_state, p=2, dim=-1) * np.sqrt(self.hidden_dim) # normalize the hidden states
        if self.obs_shortcut:
            joint_embed = torch.cat((obs_embeds[-1], hidden_state), dim = -1)
        else:
            joint_embed = hidden_state


        return joint_embed, current_internal_state
