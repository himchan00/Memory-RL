import os
import torchkit.pytorch_utils as ptu
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, obs_backend="ram", obs_dtype="float32", memmap_dir=None, max_seq_len=-1):
        # If action_dim is None, we are dealing with discrete actions
        if action_dim is None:
            action_dim = 1
            self.act_continuous = False
        else:
            self.act_continuous = True
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.sampled_seq_len = max_episode_len + 1 # +1 for dummy step at t = -1
        # Training BPTT window (number of real transitions). Full episodes are
        # always stored; when 0 < max_seq_len < max_episode_len, random_episodes
        # samples a random contiguous window of (max_seq_len + 1) rows per item
        # and marks its first row as the dummy/reset step (mask=0). Models thus
        # reset their memory at the window start (amago-style truncated BPTT).
        # Inference is unaffected (full history is used at rollout time).
        self.window_enabled = 0 < max_seq_len < max_episode_len
        self.max_seq_len = max_seq_len
        self.num_episodes = num_episodes
        self.obs_backend = obs_backend
        self.obs_dtype = np.dtype(obs_dtype)
        self.memmap_dir = memmap_dir
        if self.obs_backend == "memmap":
            assert self.memmap_dir is not None, "memmap_dir required when obs_backend='memmap'"
            os.makedirs(self.memmap_dir, exist_ok=True)
            self._obs_path = os.path.join(self.memmap_dir, "obs.dat")
            self._obs2_path = os.path.join(self.memmap_dir, "next_obs.dat")
            size_gb = (num_episodes * (max_episode_len + 1) * observation_dim
               * self.obs_dtype.itemsize * 2) / (1024 ** 3)
            print(f"[RolloutBuffer] memmap backend: {size_gb:.2f} GB across "
                f"{self._obs_path} and {self._obs2_path}")
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
        return each item has 3D shape (L, batch_size, dim), where
        L = sampled_seq_len (full episode) or, when windowing is enabled,
        L = max_seq_len + 1 (a random contiguous BPTT window per item).
        """
        sampled_indices = self._sample_indices(batch_size)
        act = self.actions[:, sampled_indices, :]
        rew = self.rewards[:, sampled_indices, :]
        if self.obs_backend == "ram":
            obs = self.observations[:, sampled_indices, :]
            obs2 = self.next_observations[:, sampled_indices, :]
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

        batch = dict(
            act=act,
            obs=obs,
            obs2=obs2,
            rew=rew,
            term=self.terminals[:, sampled_indices, :],
            mask=self.masks[:, sampled_indices, :],
        )
        if self.window_enabled:
            batch = self._window(batch, batch_size)
        return batch

    def _window(self, batch, batch_size):
        """Slice a random contiguous window of (max_seq_len + 1) rows per item.

        Each item b gets an independent start s_b in [0, sampled_seq_len - (L+1)];
        rows [s_b : s_b + L + 1] are gathered along the time axis. The window's
        first row is forced to be the dummy/reset step (mask=0) so every seq model
        starts fresh at the window boundary.
        """
        win_rows = self.max_seq_len + 1                 # dummy + L real transitions
        max_start = self.sampled_seq_len - win_rows     # inclusive upper bound
        starts = torch.randint(0, max_start + 1, (batch_size,), device=ptu.device)
        ar = torch.arange(win_rows, device=ptu.device).unsqueeze(1)  # (win_rows, 1)
        gather_idx = (starts.unsqueeze(0) + ar)                       # (win_rows, B)

        def _win(x):
            index = gather_idx.unsqueeze(-1).expand(win_rows, batch_size, x.shape[-1])
            return torch.gather(x, 0, index)

        out = {k: _win(v) for k, v in batch.items()}
        out["mask"] = out["mask"].clone()
        out["mask"][0] = 0.0  # window start acts as the t=-1 dummy (reset point)
        return out


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

    def close(self):
        if self.obs_backend == "memmap":
            if hasattr(self, "observations") and self.observations is not None:
                self.observations.flush()
                del self.observations
            if hasattr(self, "next_observations") and self.next_observations is not None:
                self.next_observations.flush()
                del self.next_observations