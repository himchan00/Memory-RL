import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import helpers as utl
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu


class Critic_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_critic,
        algo,
        image_encoder=None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo

        self.obs_shortcut = config_seq.obs_shortcut
        self.full_transition = config_seq.full_transition
        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder is None:
            observ_embedding_size = config_seq.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = image_encoder
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, config_seq.action_embedder.hidden_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(
            1, config_seq.reward_embedder.hidden_size, F.relu
        )

        ## 2. build RNN model
        observ_hidden_size = 2 * observ_embedding_size if self.full_transition else observ_embedding_size
        rnn_input_size = (
            observ_hidden_size
            + config_seq.action_embedder.hidden_size
            + config_seq.reward_embedder.hidden_size
        )
        self.seq_model = SEQ_MODELS[config_seq.seq_model_config.name](
            input_size=rnn_input_size, **config_seq.seq_model_config.to_dict()
        )

        ## 3. build q networks
        self.hidden_dim = self.seq_model.hidden_size
        input_size = self.hidden_dim
        if self.obs_shortcut:
            input_size += obs_dim
        if self.algo.continuous_action:
            input_size += action_dim
        qf = self.algo.build_critic(
            input_size=input_size,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )
        if isinstance(qf, tuple):
            self.qf1, self.qf2 = qf
        else:
            self.qf = qf

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None
    ):
        """
        Inputs:
        actions[t] = a_t, shape (L, B, dim)
        rewards[t] = r_t, shape (L, B, dim)
        observs[t] = o_t, shape (L+1, B, dim)
        Outputs:
        hidden[t] = h_t: (L, B, dim)
        """
        observs_t = observs[:-1] # o[t]
        observs_t_1 = observs[1:] # o[t+1]
        input_a = self.action_embedder(actions)
        input_r = self.reward_embedder(rewards)
        if self.full_transition:
            input_s = self.observ_embedder(observs_t)
            input_s_1 = self.observ_embedder(observs_t_1)
            inputs = torch.cat((input_a, input_r, input_s, input_s_1), dim=-1)
        elif self.obs_shortcut:
            input_s = self.observ_embedder(observs_t)
            inputs = torch.cat((input_a, input_r, input_s), dim=-1)
        else: 
            input_s_1 = self.observ_embedder(observs_t_1)
            inputs = torch.cat((input_a, input_r, input_s_1), dim=-1)

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1]
            )  # initial_internal_state is zeros
            output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
            return output, current_internal_state

    def forward(self, actions, rewards, observs, current_actions):
        """
        Inputs:
        actions[t] = a_t, shape (L, B, dim)
        rewards[t] = r_t, shape (L, B, dim)
        observs[t] = o_t, shape (L+1, B, dim)
        current_actions[t] = a'_t, shape (L+1, B, dim) or None
        Outputs:
        Q_values[t] = Q(o_{0:t}, a_{0:t-1}, r_{0:t-1}), shape (L+1, B, dim)
        """
        assert actions.dim() == rewards.dim() == observs.dim() == 3
        assert actions.shape[0] + 1 == rewards.shape[0] + 1  == observs.shape[0]
        bs = actions.shape[1]
        if not self.obs_shortcut:
            o, a, r = ptu.zeros((1, bs, self.obs_dim)).float(), ptu.zeros((1, bs, self.action_dim)).float(), ptu.zeros((1, bs, 1)).float() 
            observs, actions, rewards = torch.cat((o, observs), dim = 0), torch.cat((a, actions), dim = 0), torch.cat((r, rewards), dim = 0)
        
        hidden_states = self.get_hidden_states(
            actions=actions, rewards=rewards, observs=observs
        )

        if self.obs_shortcut:
            h = ptu.zeros((1, bs, self.hidden_dim)).float()
            hidden_states = torch.cat((h, hidden_states), dim = 0)

        joint_embeds = hidden_states # (L+1, B, dim)

        if self.obs_shortcut:
            joint_embeds = torch.cat((observs, joint_embeds), dim = -1) # Q(s, h)

        if self.algo.continuous_action:
            joint_embeds = torch.cat((joint_embeds, current_actions), dim = -1) # Q(s, h, a)

        # q value
        if hasattr(self, "qf"):
            q = self.qf(joint_embeds)
            return q
        else:
            q1 = self.qf1(joint_embeds)
            q2 = self.qf2(joint_embeds)
            return q1, q2  # (T, B, 1 or A)

    @torch.no_grad()
    def get_initial_info(self, max_attn_span: int = -1):
        prev_obs = ptu.zeros((1, self.obs_dim)).float()
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.seq_model.get_zero_internal_state()

        return prev_obs, prev_action, reward, internal_state

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        prev_reward,
        prev_obs,
        obs,
        deterministic=False,
        initial=False
    ):
        """
        Used for evaluation (not training) so L=1
        prev_action a_{t-1}, (1, B, dim) 
        prev_reward r_{t-1}, (1, B, 1)
        prev_obs o_{t-1}, (1, B, dim)
        obs o_{t} (1, B, dim) 
        Note: When initial=True, prev_ data are not used
        """
        assert prev_action.dim() == prev_reward.dim() == prev_obs.dim() == obs.dim() == 3
        if initial and self.obs_shortcut:
            hidden_state = ptu.zeros((1, self.hidden_dim)).float()
            current_internal_state = self.seq_model.get_zero_internal_state()
        else:
            if initial and not self.obs_shortcut:
                prev_obs, prev_action, prev_reward = ptu.zeros((1, 1, self.obs_dim)).float(), ptu.zeros((1, 1, self.action_dim)).float(), ptu.zeros((1, 1, 1)).float() 
                prev_internal_state = self.seq_model.get_zero_internal_state()
                
            hidden_state, current_internal_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=torch.cat((prev_obs, obs), dim = 0),
                initial_internal_state=prev_internal_state,
            )
            hidden_state = hidden_state.squeeze(0)  # (B, dim)

        joint_embeds = hidden_state
        if self.obs_shortcut:
            joint_embeds = torch.cat((obs.squeeze(0), joint_embeds), dim = -1)

        current_action = self.algo.select_action(
            qf=self.qf,  # assume single q head
            observ=joint_embeds,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
