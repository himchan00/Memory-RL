import torch
import torch.nn as nn
import numpy as np
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp, double_Mlp


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
        if self.project_output:
            self.init_emb = nn.Parameter(ptu.randn(self.hidden_dim))
        self.permutation_training = config_seq.get("permutation_training", False)
        self.transition_dropout = config_seq.get("transition_dropout", 0.0)
        self.is_target = None
        if self.permutation_training: # These will be set externally during training
            assert (self.obs_shortcut and config_seq.seq_model.name == "mate"), "Permutation training is only implemented for Mate with obs_shortcut=True"
            # Permutation indices for transition and memory
            self.transition_perm = None
            self.memory_perm = None
        if self.transition_dropout > 0.0:
            assert config_seq.seq_model.name == "mate", "Transition dropout is only implemented for Mate"
            assert self.permutation_training == False, "Transition dropout is not compatible with permutation training"
            self.dropout_indices = None  # These will be set externally during training

        print(f"Sequence model options: obs_shortcut={self.obs_shortcut}, full_transition={self.full_transition}, project_output={self.project_output}")
        ### Build Model
        ## 1. Observation embedder, Transition embedder
        if self.obs_shortcut:
            if config_seq.seq_model.name == "markov" and config_seq.seq_model.is_oracle:
                context_dim = config_seq.seq_model.context_dim
                true_obs_dim = obs_dim - context_dim
                self.observ_embedder = double_Mlp(Mlp(input_size=true_obs_dim, output_size=self.hidden_dim, **config_seq.embedder.to_dict()), 
                                                   Mlp(input_size=context_dim, output_size=self.hidden_dim, **config_seq.embedder.to_dict()))
            else:
                input_size = obs_dim
                self.observ_embedder = Mlp(input_size=input_size, output_size=self.hidden_dim,**config_seq.embedder.to_dict())
        else:
            self.observ_embedder = None

        transition_size = 2 * self.obs_dim + action_dim + 1 if self.full_transition else self.obs_dim + action_dim + 1
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

        ## 3. Set embedding size
        if config_seq.seq_model.name == "markov":
            self.hidden_dim = 0  # no hidden state for Markov model
        self.embedding_size = self.hidden_dim
        if self.obs_shortcut:
            self.embedding_size += self.observ_embedder.output_size
        

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None
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
                h0 = ptu.zeros((1, inputs.shape[1], self.hidden_dim)).float()
                if self.permutation_training:
                    trans_emb = self.seq_model.embedder(inputs) # (T, B, dim)
                    transition_idx = self.transition_perm.unsqueeze(-1).expand(-1, -1, self.hidden_dim)  # (T, B, dim)
                    trans_emb_perm = trans_emb.gather(0, transition_idx)  # (T, B, dim)
                    mem_emb = torch.cat((h0, trans_emb_perm.cumsum(dim = 0)), dim=0)  # (T+1, B, dim)
                    memory_idx = self.memory_perm.unsqueeze(-1).expand(-1, -1, self.hidden_dim)  # (T+1, B, dim)
                    mem_emb_perm = mem_emb.gather(0, memory_idx)  # (T+1, B, dim)
                    if self.is_target: # target joint emb: (s_{t+1}, m_{t} + trans(x_{t+1}))
                        mem_emb_perm[1:] = mem_emb_perm[:-1].clone() + trans_emb
                    output = mem_emb_perm
                if self.transition_dropout > 0.0:
                    trans_emb = self.seq_model.embedder(inputs) # (T, B, dim)
                    trans_emb_dropped = trans_emb * self.dropout_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
                    output = torch.cat((h0, trans_emb_dropped.cumsum(dim = 0)), dim=0)  # (T+1, B, dim)
                    if self.is_target:
                        output[1:] = output[:-1].clone() + trans_emb
                else:
                    output, _ = self.seq_model(inputs, initial_internal_state)
                    output = torch.cat((h0, output), dim = 0) # add zero hidden state at t = 0

            else:
                output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
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
        
        hidden_states = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs
        )  # (T+1, B, dim)
        h_dummy = ptu.zeros((1, hidden_states.shape[1], hidden_states.shape[2])).float()
        hidden_states = torch.cat((h_dummy, hidden_states), dim = 0) # (T+2, B, dim), add zero hidden state at t = -1 for alignment with observs
        if self.project_output:
            hidden_states = hidden_states + self.init_emb
            hidden_states = hidden_states / hidden_states.norm(dim = -1, keepdim=True).clamp(min=1e-6) * np.sqrt(self.hidden_dim) # normalize the hidden states
        if self.obs_shortcut:
            observs_embeds = self.observ_embedder(observs) 
            joint_embeds = torch.cat((observs_embeds, hidden_states), dim = -1) # Q(s, h)
        else:
            joint_embeds = hidden_states # Q(h)

        if self.seq_model.hidden_size > 0: 
            d_forward = {"hidden_states_mean": hidden_states.detach().mean(dim = (1, 2)),
                        "hidden_states_std": hidden_states.detach().std(dim = 2).mean(dim = 1)}
        else: # To avoid warning when using Markov policy
            d_forward = {}
        if self.obs_shortcut:
            d_forward["observs_embeds_mean"], d_forward["observs_embeds_std"] = observs_embeds.detach().mean(dim = (1, 2)), observs_embeds.detach().std(dim = 2).mean(dim = 1)

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
        if initial and self.obs_shortcut:
            current_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state = ptu.zeros((1, bs, self.hidden_dim)).float()
        else:
            if initial:
                prev_internal_state = self.seq_model.get_zero_internal_state(batch_size=bs)
            hidden_state, current_internal_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=torch.cat((prev_obs, obs), dim = 0),
                initial_internal_state=prev_internal_state,
            )
        hidden_state = hidden_state.squeeze(0)  # (B, dim)
        if self.project_output:
            hidden_state = hidden_state + self.init_emb
            hidden_state = hidden_state / hidden_state.norm(dim = -1, keepdim=True).clamp(min=1e-6) * np.sqrt(self.hidden_dim) # normalize the hidden states
        if self.obs_shortcut:
            obs_embed = self.observ_embedder(obs) 
            joint_embed = torch.cat((obs_embed.squeeze(0), hidden_state), dim = -1)
        else:
            joint_embed = hidden_state


        return joint_embed, current_internal_state
