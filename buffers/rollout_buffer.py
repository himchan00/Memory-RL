import os
import torchkit.pytorch_utils as ptu
import torch
import numpy as np
from utils.helpers import RunningMeanStd

class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, normalize_transitions, obs_backend="ram", obs_dtype="float32", memmap_dir=None):
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
        self.obs_backend = obs_backend
        self.obs_dtype = np.dtype(obs_dtype)
        self.memmap_dir = memmap_dir
        if self.obs_backend == "memmap":
            assert self.memmap_dir is not None, "memmap_dir required when obs_backend='memmap'"
            assert not self.normalize_transitions, (
                "normalize_transitions must be False when obs_backend='memmap' "
                "(RunningMeanStd expects float obs; images are stored as uint8)."
                )
            os.makedirs(self.memmap_dir, exist_ok=True)
            self._obs_path = os.path.join(self.memmap_dir, "obs.dat")
            self._obs2_path = os.path.join(self.memmap_dir, "next_obs.dat")
            size_gb = (num_episodes * (max_episode_len + 1) * observation_dim
               * self.obs_dtype.itemsize * 2) / (1024 ** 3)
            print(f"[RolloutBuffer] memmap backend: {size_gb:.2f} GB across "
                f"{self._obs_path} and {self._obs2_path}")
        if self.normalize_transitions:
            self.observation_rms = RunningMeanStd(shape=(self.observation_dim,))
            self.rewards_rms = RunningMeanStd(shape=(1,))
        self.reset()


    def reset(self):
        self.actions = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.action_dim))
        if not self.act_continuous:
            self.actions = self.actions.long() # dtype for discrete actions are long
        if self.obs_backend == "ram":
            self.observations = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.observation_dim))
            self.next_observations = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.observation_dim))
        else:  # memmap
            shape = (self.num_episodes, self.sampled_seq_len, self.observation_dim)
            self.observations = np.memmap(self._obs_path, dtype=self.obs_dtype, mode="w+", shape=shape)
            self.next_observations = np.memmap(self._obs2_path, dtype=self.obs_dtype, mode="w+", shape=shape)
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
        self.rewards[:, indices, :] = rewards.detach()
        self.terminals[:, indices, :] = terminals.detach()
        if self.obs_backend == "ram":            
            self.observations[:, indices, :] = observations.detach()
            self.next_observations[:, indices, :] = next_observations.detach()
        else:
            obs_np = observations.detach().cpu().numpy()
            obs2_np = next_observations.detach().cpu().numpy()
            if np.issubdtype(self.obs_dtype, np.integer):
                obs_np = np.clip(obs_np, 0, np.iinfo(self.obs_dtype).max)
                obs2_np = np.clip(obs2_np, 0, np.iinfo(self.obs_dtype).max)
            obs_np = np.transpose(obs_np, (1, 0, 2)).astype(self.obs_dtype, copy=False)
            obs2_np = np.transpose(obs2_np, (1, 0, 2)).astype(self.obs_dtype, copy=False)
            for i, s in enumerate(indices):
                self.observations[s] = obs_np[i]
                self.next_observations[s] = obs2_np[i]
        
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
        rew_raw = self.rewards[:, sampled_indices, :]
        if self.obs_backend == "ram":
            obs_raw = self.observations[:, sampled_indices, :]
            obs2_raw = self.next_observations[:, sampled_indices, :]
            if self.normalize_transitions:
                obs = self.observation_rms.norm(obs_raw)
                obs2 = self.observation_rms.norm(obs2_raw)
                rew = self.rewards_rms.norm(rew_raw, scale=False)
            else:
                obs = obs_raw
                obs2 = obs2_raw
                rew = rew_raw
        else:
            idx_np = sampled_indices.detach().cpu().numpy()
            # (B, T+1, D) -> float32 GPU tensor, then permute to (T+1, B, D)
            obs_np = np.ascontiguousarray(self.observations[idx_np])
            obs2_np = np.ascontiguousarray(self.next_observations[idx_np])
            obs = torch.from_numpy(obs_np).to(
                ptu.device, dtype=torch.float32, non_blocking=True
            ).permute(1, 0, 2).contiguous()
            obs2 = torch.from_numpy(obs2_np).to(
                ptu.device, dtype=torch.float32, non_blocking=True
            ).permute(1, 0, 2).contiguous()
            rew = rew_raw  # reward normalization disabled with memmap (see __init__)
            
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
            "rewards": self.rewards.cpu(),
            "terminals": self.terminals.cpu(),
            "masks": self.masks.cpu(),
            "valid_index": self.valid_index.cpu(),
            "_top": self._top,
            "obs_backend": self.obs_backend,
        }
        if self.obs_backend == "ram":
            d["observations"] = self.observations.cpu()
            d["next_observations"] = self.next_observations.cpu()
        else:
            self.observations.flush()
            self.next_observations.flush()
            d["memmap_dir"] = self.memmap_dir
            d["obs_dtype"] = str(self.obs_dtype)
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
        self.rewards = state_dict["rewards"].to(ptu.device)
        self.terminals = state_dict["terminals"].to(ptu.device)
        self.masks = state_dict["masks"].to(ptu.device)
        self.valid_index = state_dict["valid_index"].to(ptu.device)
        self._top = state_dict["_top"]
        
        saved_backend = state_dict.get("obs_backend", "ram")
        assert saved_backend == self.obs_backend, (f"Saved obs_backend {saved_backend} does not match current obs_backend {self.obs_backend}")
        if self.obs_backend == "ram":
            self.observations = state_dict["observations"].to(ptu.device)
            self.next_observations = state_dict["next_observations"].to(ptu.device)
        else:
            shape = (self.num_episodes, self.sampled_seq_len, self.observation_dim)
            self.observations = np.memmap(self._obs_path, dtype=self.obs_dtype, mode="r+", shape=shape)
            self.next_observations = np.memmap(self._obs2_path, dtype=self.obs_dtype, mode="r+", shape=shape)
            
        if self.normalize_transitions:
            self.observation_rms.mean = state_dict["observation_rms_mean"]
            self.observation_rms.var = state_dict["observation_rms_var"]
            self.observation_rms.count = state_dict["observation_rms_count"]
            self.rewards_rms.mean = state_dict["rewards_rms_mean"]
            self.rewards_rms.var = state_dict["rewards_rms_var"]
            self.rewards_rms.count = state_dict["rewards_rms_count"]
            
    def close(self):
        if self.obs_backend == "memmap":
            if hasattr(self, "observations") and self.observations is not None:
                self.observations.flush()
                del self.observations
            if hasattr(self, "next_observations") and self.next_observations is not None:
                self.next_observations.flush()
                del self.next_observations