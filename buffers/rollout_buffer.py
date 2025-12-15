import torchkit.pytorch_utils as ptu
import torch
import numpy as np
from utils.helpers import RunningMeanStd

class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, normalize_transitions, is_ppo = False):
        # If action_dim is None, we are dealing with discrete actions
        if action_dim is None:
            action_dim = 1
            self.act_continuous = False
        else:
            self.act_continuous = True
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.sampled_seq_len = max_episode_len + 1 # +1 for dummy step at t = -1
        self.num_episodes = num_episodes
        self.is_ppo = is_ppo
        self.normalize_transitions = normalize_transitions
        print(f"Normalize transitions: {self.normalize_transitions}")
        if self.normalize_transitions:
            self.observation_rms = RunningMeanStd(shape=(self.observation_dim,))
            self.rewards_rms = RunningMeanStd(shape=(1,))
        self.reset()


    def reset(self):
        self.actions = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.action_dim))
        if not self.act_continuous:
            self.actions = self.actions.long() # dtype for discrete actions are long
        self.observations = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.observation_dim))
        self.next_observations = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.observation_dim))
        self.rewards = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.terminals = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.masks = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.valid_index = ptu.zeros((self.num_episodes))

        if self.is_ppo:
            self.values = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
            self.logprobs = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
            self.advantages = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
            self.returns = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))

        self._top = 0



    def add_episode(self, actions, observations, next_observations, rewards, terminals, values=None, logprobs=None):
        """
        All inputs are of the shape (T+1, B, ...)
        """
        seq_len = actions.shape[0]
        batch_size = actions.shape[1]
        assert observations.shape[0] == next_observations.shape[0] == rewards.shape[0] == terminals.shape[0] == seq_len
        assert observations.shape[1] == next_observations.shape[1] == rewards.shape[1] == terminals.shape[1] == batch_size
        if values is not None:
            assert (values.shape[0], values.shape[1]) == (seq_len, batch_size)
        if logprobs is not None:
            assert (logprobs.shape[0], logprobs.shape[1]) == (seq_len, batch_size)

        if self.normalize_transitions: # do not use dummy step at t = -1
            self.observation_rms.update(observations[1:]) 
            self.rewards_rms.update(rewards[1:])

        indices = list(
            np.arange(self._top, self._top + batch_size) % self.num_episodes
        )
        self.actions[:, indices, :] = actions.detach()
        self.observations[:, indices, :] = observations.detach()
        self.next_observations[:, indices, :] = next_observations.detach()
        self.rewards[:, indices, :] = rewards.detach()
        self.terminals[:, indices, :] = terminals.detach()
        masks = ptu.ones_like(terminals)
        masks[0] = 0.0  # mask at t = -1 is 0
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
        act_raw = self.actions[:, sampled_indices, :]
        obs_raw = self.observations[:, sampled_indices, :]
        obs2_raw = self.next_observations[:, sampled_indices, :]
        rew_raw = self.rewards[:, sampled_indices, :]
        if self.normalize_transitions:
            obs = self.observation_rms.norm(obs_raw)
            obs2 = self.observation_rms.norm(obs2_raw)
            rew = self.rewards_rms.norm(rew_raw, scale=False)
        else:
            obs = obs_raw
            obs2 = obs2_raw
            rew = rew_raw
        return dict(
            act=act_raw,
            obs=obs,
            obs2=obs2,
            rew=rew,
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


    def state_dict(self):
        state_dict = {
            "observation_rms_mean": self.observation_rms.mean,
            "observation_rms_var": self.observation_rms.var,
            "observation_rms_count": self.observation_rms.count,
            "rewards_rms_mean": self.rewards_rms.mean,
            "rewards_rms_var": self.rewards_rms.var,
            "rewards_rms_count": self.rewards_rms.count,
        }
        return state_dict
    
    
    def load_state_dict(self, state_dict):
        self.observation_rms.mean = state_dict["observation_rms_mean"]
        self.observation_rms.var = state_dict["observation_rms_var"]
        self.observation_rms.count = state_dict["observation_rms_count"]
        self.rewards_rms.mean = state_dict["rewards_rms_mean"]
        self.rewards_rms.var = state_dict["rewards_rms_var"]
        self.rewards_rms.count = state_dict["rewards_rms_count"]