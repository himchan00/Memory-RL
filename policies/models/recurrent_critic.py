import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_head import RNN_head


class Critic_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_critic,
        algo,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.algo = algo

        self.head = RNN_head(
            obs_dim,
            action_dim,
            config_seq,
        )
        self.qf = self.algo.build_critic(
            input_size=self.head.embedding_size,
            hidden_sizes=config_critic.hidden_dims,
            action_dim=action_dim,
        )


    def forward(self, actions, rewards, observs):
        """
        Inputs:
        actions[t] = a_t, shape (L, B, dim)
        rewards[t] = r_t, shape (L, B, dim)
        observs[t] = o_t, shape (L+1, B, dim)
        Outputs:
        Q_values[t] = Q(o_{0:t}, a_{0:t-1}, r_{0:t-1}), shape (L+1, B, dim)
        """
        assert actions.dim() == rewards.dim() == observs.dim() == 3
        assert actions.shape[0] + 1 == rewards.shape[0] + 1  == observs.shape[0]
        
        joint_embeds, d_forward = self.head.forward(actions=actions, rewards=rewards, observs=observs)

        q = self.qf(joint_embeds)
        return q, d_forward

    @torch.no_grad()
    def get_initial_info(self, batch_size):
        prev_obs = ptu.zeros((batch_size, self.obs_dim)).float()
        prev_action = ptu.zeros((batch_size, self.action_dim)).float()
        reward = ptu.zeros((batch_size, 1)).float()
        internal_state = self.head.seq_model.get_zero_internal_state(batch_size=batch_size)

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

        joint_embed, current_internal_state = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial
        )

        current_action = self.algo.select_action(
            qf=self.qf,  # assume single q head
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
