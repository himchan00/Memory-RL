import torchkit.pytorch_utils as ptu
import torch
import numpy as np
from utils.helpers import RunningMeanStd

class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, normalize_transitions):
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

        self._top = 0



    def add_episode(self, actions, observations, next_observations, rewards, terminals):
        """
        All inputs are of the shape (T+1, B, ...)
        """
        seq_len = actions.shape[0]
        batch_size = actions.shape[1]
        assert observations.shape[0] == next_observations.shape[0] == rewards.shape[0] == terminals.shape[0] == seq_len
        assert observations.shape[1] == next_observations.shape[1] == rewards.shape[1] == terminals.shape[1] == batch_size

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
    




    def state_dict(self):
        d = {
            "actions": self.actions.cpu(),
            "observations": self.observations.cpu(),
            "next_observations": self.next_observations.cpu(),
            "rewards": self.rewards.cpu(),
            "terminals": self.terminals.cpu(),
            "masks": self.masks.cpu(),
            "valid_index": self.valid_index.cpu(),
            "_top": self._top,
        }
        if self.normalize_transitions:
            d["observation_rms_mean"] = self.observation_rms.mean
            d["observation_rms_var"] = self.observation_rms.var
            d["observation_rms_count"] = self.observation_rms.count
            d["rewards_rms_mean"] = self.rewards_rms.mean
            d["rewards_rms_var"] = self.rewards_rms.var
            d["rewards_rms_count"] = self.rewards_rms.count
        return d

    def load_state_dict(self, state_dict):
        self.actions = state_dict["actions"].to(ptu.device)
        if not self.act_continuous:
            self.actions = self.actions.long()
        self.observations = state_dict["observations"].to(ptu.device)
        self.next_observations = state_dict["next_observations"].to(ptu.device)
        self.rewards = state_dict["rewards"].to(ptu.device)
        self.terminals = state_dict["terminals"].to(ptu.device)
        self.masks = state_dict["masks"].to(ptu.device)
        self.valid_index = state_dict["valid_index"].to(ptu.device)
        self._top = state_dict["_top"]
        if self.normalize_transitions:
            self.observation_rms.mean = state_dict["observation_rms_mean"]
            self.observation_rms.var = state_dict["observation_rms_var"]
            self.observation_rms.count = state_dict["observation_rms_count"]
            self.rewards_rms.mean = state_dict["rewards_rms_mean"]
            self.rewards_rms.var = state_dict["rewards_rms_var"]
            self.rewards_rms.count = state_dict["rewards_rms_count"]