import torchkit.pytorch_utils as ptu
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, is_ppo = False):
        # If action_dim is None, we are dealing with discrete actions
        if action_dim is None:
            action_dim = 1
            self.act_continuous = False
        else:
            self.act_continuous = True
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.max_episode_len = max_episode_len
        self.num_episodes = num_episodes
        self.is_ppo = is_ppo
        self.reset()


    def reset(self):
        self.actions = ptu.zeros((self.max_episode_len, self.num_episodes, self.action_dim))
        if not self.act_continuous:
            self.actions = self.actions.long() # dtype for discrete actions are long
        self.observations = ptu.zeros((self.max_episode_len, self.num_episodes, self.observation_dim))
        self.next_observations = ptu.zeros((self.max_episode_len, self.num_episodes, self.observation_dim))
        self.rewards = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
        self.terminals = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
        self.masks = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
        self.valid_index = ptu.zeros((self.num_episodes))

        if self.is_ppo:
            self.values = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
            self.logprobs = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
            self.advantages = ptu.zeros((self.max_episode_len, self.num_episodes, 1))
            self.returns = ptu.zeros((self.max_episode_len, self.num_episodes, 1))

        self._top = 0



    def add_episode(self, actions, observations, next_observations, rewards, terminals, values=None, logprobs=None):
        """
        All inputs are of the shape (T, B, ...)
        """
        seq_len = actions.shape[0]
        batch_size = actions.shape[1]
        assert observations.shape[0] == next_observations.shape[0] == rewards.shape[0] == terminals.shape[0] == seq_len
        assert observations.shape[1] == next_observations.shape[1] == rewards.shape[1] == terminals.shape[1] == batch_size
        if values is not None:
            assert (values.shape[0], values.shape[1]) == (seq_len, batch_size)
        if logprobs is not None:
            assert (logprobs.shape[0], logprobs.shape[1]) == (seq_len, batch_size)

        indices = list(
            np.arange(self._top, self._top + batch_size) % self.num_episodes
        )
        self.actions[:, indices, :] = actions.detach()
        self.observations[:, indices, :] = observations.detach()
        self.next_observations[:, indices, :] = next_observations.detach()
        self.rewards[:, indices, :] = rewards.detach()
        self.terminals[:, indices, :] = terminals.detach()
        masks = ptu.ones_like(terminals)
        masks[1:] = (1-terminals[:-1])
        self.masks[:, indices, :] = masks.detach()
        self.valid_index[indices] = 1.0
        if values is not None:
            self.values[:, indices, :] = values.detach()
        if logprobs is not None:
            self.logprobs[:, indices, :] = logprobs.detach()
        self._top += batch_size


    def random_episodes(self, batch_size):
        """
        return each item has 3D shape (sampled_seq_len, batch_size, dim)
        Note: This simplified implementation assumes that sampled_seq_len = self.max_episode_len
        """
        sampled_indices = self._sample_indices(batch_size)
        return dict(
            act=self.actions[:, sampled_indices, :],
            obs=self.observations[:, sampled_indices, :],
            obs2=self.next_observations[:, sampled_indices, :],
            rew=self.rewards[:, sampled_indices, :],
            term=self.terminals[:, sampled_indices, :],
            mask=self.masks[:, sampled_indices, :],
        )


    def _sample_indices(self, batch_size):
        valid_indices = torch.where(self.valid_index > 0.0)[0]

        sample_weights = torch.clone(self.valid_index[valid_indices])
        # normalize to probability distribution
        sample_weights /= sample_weights.sum()

        return torch.multinomial(sample_weights, num_samples=batch_size, replacement=True)
    

    def compute_gae(self, gamma, lam):
        """
        Note: this code does not bootstrap final value.
        This is allowed for finite horizon tasks.
        """
        B, T = self.num_episodes, self.max_episode_len

        rewards   = self.rewards.squeeze(-1)    # (T, B)
        values    = self.values.squeeze(-1)     # (T, B)
        terminals = self.terminals.squeeze(-1)  # (T, B)  in {0,1}
        
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]      # V_{t+1}

        # TD residuals (delta_t)
        # delta_t = r_t + gamma * (1 - done_t) * V_{t+1} - V_t
        deltas = rewards + gamma * (1.0 - terminals) * next_values - values # (T, B)

        advantages = torch.zeros_like(values)
        gae = ptu.zeros((B,)) 

        for t in reversed(range(T)):
            gae = deltas[t, :] + gamma * lam * (1.0 - terminals[t, :]) * gae   # (B, 1)
            advantages[t, :] = gae

        returns = advantages + values
        self.advantages = advantages.unsqueeze(-1)
        self.returns = returns.unsqueeze(-1)
        return returns, advantages
