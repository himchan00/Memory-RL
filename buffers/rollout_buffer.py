import torchkit.pytorch_utils as ptu
import torch
import numpy as np

from buffers.observation_store import (
    MemmapObservationStore,
    RamObservationStore,
)


class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, obs_backend="ram", obs_dtype="float32", memmap_dir=None, max_seq_len=-1, require_memory_masks=False):
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
        self.require_memory_masks = require_memory_masks
        self.num_episodes = num_episodes
        self.obs_backend = obs_backend
        self.obs_dtype = np.dtype(obs_dtype)
        self.memmap_dir = memmap_dir
        if self.obs_backend not in {"ram", "memmap"}:
            raise ValueError(
                f"Unknown obs_backend {self.obs_backend!r}; expected 'ram' or 'memmap'"
            )
        if self.obs_backend == "memmap" and self.memmap_dir is None:
            raise ValueError("memmap_dir is required when obs_backend='memmap'")
        self.reset()


    def reset(self):
        if hasattr(self, "_observation_store"):
            self._observation_store.close()
        self.actions = ptu.zeros((self.sampled_seq_len, self.num_episodes, self.action_dim))
        if not self.act_continuous:
            self.actions = self.actions.long() # dtype for discrete actions are long
        self._observation_store = self._create_observation_store()
        self.rewards = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.terminals = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.masks = ptu.zeros((self.sampled_seq_len, self.num_episodes, 1))
        self.memory_masks = ptu.ones((self.sampled_seq_len, self.num_episodes, 1))
        self.valid_index = ptu.zeros((self.num_episodes))

        self._top = 0

    @property
    def observations(self):
        return self._observation_store.observations

    @property
    def next_observations(self):
        return self._observation_store.next_observations

    def _create_observation_store(self):
        common = {
            "sampled_seq_len": self.sampled_seq_len,
            "num_episodes": self.num_episodes,
            "observation_dim": self.observation_dim,
        }
        if self.obs_backend == "ram":
            return RamObservationStore(**common)

        store = MemmapObservationStore(
            **common,
            dtype=self.obs_dtype,
            directory=self.memmap_dir,
        )
        print(
            f"[RolloutBuffer] memmap backend: {store.size_gb:.2f} GB across "
            f"{store.obs_path} and {store.next_obs_path}"
        )
        return store



    def add_episode(self, actions, observations, next_observations, rewards, terminals, memory_masks):
        """
        All inputs are of the shape (T+1, B, ...)
        """
        seq_len = actions.shape[0]
        batch_size = actions.shape[1]
        assert observations.shape[0] == next_observations.shape[0] == rewards.shape[0] == terminals.shape[0] == memory_masks.shape[0] == seq_len
        assert observations.shape[1] == next_observations.shape[1] == rewards.shape[1] == terminals.shape[1] == memory_masks.shape[1] == batch_size

        indices = list(
            np.arange(self._top, self._top + batch_size) % self.num_episodes
        )
        self.actions[:, indices, :] = actions.detach()
        self.rewards[:, indices, :] = rewards.detach()
        self.terminals[:, indices, :] = terminals.detach()
        self.memory_masks[:, indices, :] = memory_masks.detach()
        self._observation_store.write(
            indices,
            observations,
            next_observations,
        )
        
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
        obs, obs2 = self._observation_store.sample(sampled_indices)

        batch = dict(
            act=act,
            obs=obs,
            obs2=obs2,
            rew=rew,
            term=self.terminals[:, sampled_indices, :],
            mask=self.masks[:, sampled_indices, :],
            memory_mask=self.memory_masks[:, sampled_indices, :],
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
        # Absolute env-step offset of each window (0 when unwindowed); lets an
        # absolute-position PE key on the true env t rather than window-relative 0.
        out["pos_offset"] = starts  # (B,)
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
            "memory_masks": self.memory_masks.cpu(),
            "valid_index": self.valid_index.cpu(),
            "_top": self._top,
            "obs_backend": self.obs_backend,
        }
        d.update(self._observation_store.state_dict())
        return d

    def load_state_dict(self, state_dict):
        self.actions = state_dict["actions"].to(ptu.device)
        if not self.act_continuous:
            self.actions = self.actions.long()
        self.rewards = state_dict["rewards"].to(ptu.device)
        self.terminals = state_dict["terminals"].to(ptu.device)
        self.masks = state_dict["masks"].to(ptu.device)
        if "memory_masks" in state_dict:
            self.memory_masks = state_dict["memory_masks"].to(ptu.device)
        elif self.require_memory_masks:
            raise ValueError(
                "This buffer checkpoint predates skip_reset_transition; "
                "start with a fresh replay buffer."
            )
        else:
            self.memory_masks = ptu.ones_like(self.masks)
        self.valid_index = state_dict["valid_index"].to(ptu.device)
        self._top = state_dict["_top"]
        
        saved_backend = state_dict.get("obs_backend", "ram")
        assert saved_backend == self.obs_backend, (f"Saved obs_backend {saved_backend} does not match current obs_backend {self.obs_backend}")
        self._observation_store.load_state_dict(state_dict)

    def close(self):
        self._observation_store.close()