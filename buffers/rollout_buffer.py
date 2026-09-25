import math

import torchkit.pytorch_utils as ptu
import torch
from torch.nn import functional as F
import numpy as np

from buffers.observation_store import (
    MemmapObservationStore,
    RamObservationStore,
)


class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, num_episodes, obs_backend="ram", obs_dtype="float32", memmap_dir=None, max_seq_len=-1, cached_embedding_dim=None, transition_sampling_method="epoch", independent_loss_rows=False, cache_ema_beta=1.0):
        # If action_dim is None, we are dealing with discrete actions
        if action_dim is None:
            action_dim = 1
            self.act_continuous = False
        else:
            self.act_continuous = True
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.sampled_seq_len = max_episode_len + 1 # +1 for dummy step at t = -1
        # Training sample length (number of real transitions). Full episodes are
        # always stored; when 0 < max_seq_len < max_episode_len, random_episodes
        # samples either a random contiguous window or a sorted random subset.
        # The first row is the context/reset step (mask=0), so sequence models
        # start fresh at the sample boundary. Inference still uses full history.
        self.max_seq_len = max_episode_len if max_seq_len <= 0 else max_seq_len
        self.num_episodes = num_episodes
        self.obs_backend = obs_backend
        self.obs_dtype = np.dtype(obs_dtype)
        self.memmap_dir = memmap_dir
        self.cached_embedding_dim = cached_embedding_dim
        # STORE subset mode: rows to re-embed follow `transition_sampling_method`;
        # with `independent_loss_rows` the loss rows are a separate iid subset.
        if transition_sampling_method not in {"epoch", "iid"}:
            raise ValueError(
                f"Unknown transition_sampling_method {transition_sampling_method!r}; expected 'epoch' or 'iid'"
            )
        self.transition_sampling_method = transition_sampling_method
        self.independent_loss_rows = bool(independent_loss_rows)
        # Refreshed embeddings enter the cache as a per-update EMA: a row last
        # written D updates ago gets cache <- lerp(cache, z, 1 - (1 - beta)^D),
        # i.e. the same timescale as an EMA stepped every update, however rarely
        # the row is revisited. beta = 1 replaces the cache.
        self.cache_ema_beta = float(cache_ema_beta)
        if not 0.0 < self.cache_ema_beta <= 1.0:
            raise ValueError("cache_ema_beta must be in (0, 1]")
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
        self.valid_index = ptu.zeros((self.num_episodes))
        self.cached_embeddings = (
            ptu.zeros((self.sampled_seq_len, self.num_episodes, self.cached_embedding_dim))
            if self.cached_embedding_dim is not None
            else None
        )
        # STORE subset sampling is epoch-style: each episode holds a random
        # permutation of its rows and hands out consecutive k-blocks, so every
        # row is re-embedded exactly once per epoch.
        self.subset_perm = (
            self._new_permutations(self.num_episodes)
            if self.cached_embeddings is not None
            else None
        )
        self.subset_cursor = (
            torch.zeros(self.num_episodes, dtype=torch.long, device=ptu.device)
            if self.cached_embeddings is not None
            else None
        )
        # Update step at which each cached row was last written (EMA only).
        self._cache_update_step = 0
        self.cache_step = (
            torch.zeros((self.sampled_seq_len, self.num_episodes), dtype=torch.long, device=ptu.device)
            if self.cached_embeddings is not None and self.cache_ema_beta < 1.0
            else None
        )

        self._top = 0

    def _new_permutations(self, n, last=None):
        # Uniform permutations of rows 1..T; rows flagged in `last` (n, T) sort last.
        keys = torch.rand(n, self.sampled_seq_len - 1, device=ptu.device)
        if last is not None:
            keys = keys + last
        return torch.argsort(keys, dim=1) + 1

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
            "dtype": self.obs_dtype,
        }
        if self.obs_backend == "ram":
            store = RamObservationStore(**common)
            print(
                f"[RolloutBuffer] ram backend: {store.size_gb:.2f} GB of "
                f"{self.obs_dtype} on {ptu.device}"
            )
            return store

        store = MemmapObservationStore(
            **common,
            directory=self.memmap_dir,
        )
        print(
            f"[RolloutBuffer] memmap backend: {store.size_gb:.2f} GB across "
            f"{store.obs_path} and {store.next_obs_path}"
        )
        return store



    def add_episode(self, actions, observations, next_observations, rewards, terminals, cached_embeddings=None):
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
        self._observation_store.write(
            indices,
            observations,
            next_observations,
        )
        if self.cached_embeddings is not None:
            self.cached_embeddings[:, indices, :] = cached_embeddings.detach()
            self.subset_perm[indices] = self._new_permutations(len(indices))
            self.subset_cursor[indices] = 0
            if self.cache_step is not None:
                self.cache_step[:, indices] = self._cache_update_step

        masks = ptu.ones_like(terminals)
        masks[0] = 0.0  # mask at t = -1 is 0
        masks[1:] = (1-terminals[:-1])
        self.masks[:, indices, :] = masks.detach()
        self.valid_index[indices] = 1.0
        self._top += batch_size


    def random_episodes(self, batch_size, mode="window"):
        """
        return each item has 3D shape (L, batch_size, dim), where
        L = sampled_seq_len (full episode) or max_seq_len + 1 when truncated.
        ``mode`` selects a contiguous BPTT ``window`` or sorted random ``subset``.
        """
        if mode not in {"window", "subset"}:
            raise ValueError(f"Unknown replay sample mode: {mode!r}")

        sampled_indices = self._sample_indices(batch_size)
        if mode == "subset":
            if self.transition_sampling_method == "epoch":
                embed_t = self._next_subset_block(sampled_indices)
            else:
                embed_t = self._random_subset(batch_size)
            loss_t = (
                self._random_subset(batch_size)
                if self.independent_loss_rows
                else embed_t
            )
            pad = embed_t.new_zeros((1, batch_size))
            transition_t = torch.cat((pad, loss_t), dim=0)
        else:
            num_rows = self.max_seq_len + 1
            max_start = self.sampled_seq_len - num_rows
            starts = (
                torch.randint(
                    0,
                    max_start + 1,
                    (batch_size,),
                    device=self.actions.device,
                )
                if max_start
                else torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=self.actions.device,
                )
            )
            transition_t = starts.unsqueeze(0) + torch.arange(
                num_rows,
                device=self.actions.device,
            ).unsqueeze(1)

        # STORE: the full-episode cache gather + prefix cumsum is computed once per batch and shared by the loss rows
        # and the re-embedded rows (it used to be recomputed inside each _materialize_rows call).
        cache_ctx = self._cache_context(sampled_indices) if self.cached_embeddings is not None else None
        batch = self._materialize_rows(sampled_indices, transition_t, cache_ctx)
        if mode == "subset":
            # Loss normalizer = expected valid rows of the sampled episodes, not
            # the random count in the subset, so the update is unbiased for the
            # full-episode update on the same episodes.
            batch["num_valid"] = self.masks[:, sampled_indices].sum() * (
                self.max_seq_len / (self.sampled_seq_len - 1)
            )
        if mode == "subset" and self.independent_loss_rows:
            # Rows whose embeddings are recomputed; same layout (row 0 = dummy).
            batch["store"] = self._materialize_rows(
                sampled_indices, torch.cat((pad, embed_t), dim=0), cache_ctx
            )
        batch["sample_mode"] = mode
        return batch

    def _cache_context(self, sampled_indices):
        """Full-episode cache of the sampled episodes and its exclusive prefix sums, (T+1, B, dim) each."""
        episode_cache = self.cached_embeddings[:, sampled_indices, :]
        prefix_before = torch.cat(
            (torch.zeros_like(episode_cache[:1]), episode_cache.cumsum(dim=0)[:-1]),
            dim=0,
        )
        return episode_cache, prefix_before

    def _materialize_rows(self, sampled_indices, transition_t, cache_ctx=None):
        episode_indices = sampled_indices.unsqueeze(0).expand_as(transition_t)

        def _gather(x):
            return x[transition_t, episode_indices, :]

        obs, obs2 = self._observation_store.sample(episode_indices, transition_t)
        mask = _gather(self.masks).clone()
        mask[0] = 0.0  # sample start acts as the t=-1 dummy (reset point)
        batch = {
            "act": _gather(self.actions),
            "rew": _gather(self.rewards),
            "term": _gather(self.terminals),
            "mask": mask,
            "obs": obs,
            "obs2": obs2,
            # Absolute replay rows for PE and shared-state normalization.
            "transition_t": transition_t,
        }
        if self.cached_embeddings is not None:
            episode_cache, prefix_before = cache_ctx if cache_ctx is not None else self._cache_context(sampled_indices)
            batch_indices = torch.arange(
                sampled_indices.shape[0],
                device=transition_t.device,
            ).unsqueeze(0).expand_as(transition_t)
            batch.update(
                {
                    "cached_embeddings": episode_cache[
                        transition_t, batch_indices, :
                    ],
                    "cached_prefixes": prefix_before[
                        transition_t, batch_indices, :
                    ],
                    "episode_indices": sampled_indices,
                }
            )
        return batch

    def _random_subset(self, batch_size):
        """Fresh uniform k-subset of rows 1..T per episode, sorted, shape (k, B)."""
        rows = torch.multinomial(
            self.masks.new_ones((batch_size, self.sampled_seq_len - 1)),
            num_samples=self.max_seq_len,
            replacement=False,
        ).T + 1
        return rows.sort(dim=0).values

    def _next_subset_block(self, episode_indices):
        """Next k rows of each episode's epoch permutation, sorted, shape (k, B)."""
        if self.subset_perm is None:
            raise RuntimeError("subset sampling requires the STORE embedding cache")
        k = self.max_seq_len
        num_rows = self.sampled_seq_len - 1
        batch_size = episode_indices.shape[0]
        cursor = self.subset_cursor[episode_indices]
        perm = self.subset_perm[episode_indices]
        # A block that overruns the end continues into a fresh permutation whose
        # tail holds the leftover rows, so no row repeats. Mask-select, no host sync.
        positions = torch.arange(num_rows, device=cursor.device)
        in_tail = (positions >= cursor.unsqueeze(1)).float()
        new_perm = self._new_permutations(
            batch_size, last=torch.zeros_like(in_tail).scatter_(1, perm - 1, in_tail)
        )
        rows = torch.cat((perm, new_perm), dim=1).gather(1, cursor.unsqueeze(1) + positions[:k])
        wrap = cursor + k > num_rows
        self.subset_perm[episode_indices] = torch.where(wrap.unsqueeze(1), new_perm, perm)
        self.subset_cursor[episode_indices] = cursor + k - wrap.long() * num_rows
        return rows.T.sort(dim=0).values

    def update_cached_embeddings(self, episode_indices, transition_t, embeddings):
        episode_grid = episode_indices.unsqueeze(0).expand_as(transition_t)
        embeddings = embeddings.detach()
        if self.cache_step is not None:
            # Called once per RL update; bump first so every row has D >= 1.
            self._cache_update_step += 1
            elapsed = self._cache_update_step - self.cache_step[transition_t, episode_grid]
            weight = -torch.expm1(elapsed.unsqueeze(-1) * math.log1p(-self.cache_ema_beta))
            embeddings = torch.lerp(
                self.cached_embeddings[transition_t, episode_grid, :],
                embeddings.to(self.cached_embeddings),
                weight.to(self.cached_embeddings),
            )
            self.cache_step[transition_t, episode_grid] = self._cache_update_step
        self.cached_embeddings[transition_t, episode_grid, :] = embeddings


    def _sample_indices(self, batch_size):
        valid_indices = torch.where(self.valid_index > 0)[0]
        chosen_positions = torch.multinomial(self.valid_index[valid_indices], num_samples=batch_size, replacement=True)
        return valid_indices[chosen_positions]
    




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
        if self.cached_embeddings is not None:
            # The STORE embedding cache is NOT checkpointed: it is (replay capacity x (T+1) x hidden) floats -- 5 GB at
            # T=1000, h=128 -- written and re-read at every evaluation, while it is recomputable from the buffer.
            # Learner.load_checkpoint re-embeds it (rebuild_cached_embeddings); only the sampling state is saved.
            d["subset_perm"] = self.subset_perm.cpu()
            d["subset_cursor"] = self.subset_cursor.cpu()
        if self.cache_step is not None:
            d["cache_step"] = self.cache_step.cpu()
            d["_cache_update_step"] = self._cache_update_step
        d.update(self._observation_store.state_dict())
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
        if self.cached_embeddings is not None:
            self.cache_needs_rebuild = "cached_embeddings" not in state_dict
            if not self.cache_needs_rebuild:  # checkpoints written before the cache was dropped from them
                self.cached_embeddings.copy_(state_dict["cached_embeddings"])
            if "subset_perm" in state_dict:  # absent in older checkpoints
                self.subset_perm.copy_(state_dict["subset_perm"])
                self.subset_cursor.copy_(state_dict["subset_cursor"])
        if self.cache_step is not None and "cache_step" in state_dict:
            self.cache_step.copy_(state_dict["cache_step"])
            self._cache_update_step = state_dict["_cache_update_step"]
        
        saved_backend = state_dict.get("obs_backend", "ram")
        assert saved_backend == self.obs_backend, (f"Saved obs_backend {saved_backend} does not match current obs_backend {self.obs_backend}")
        self._observation_store.load_state_dict(state_dict)

    @torch.no_grad()
    def rebuild_cached_embeddings(self, embed_transitions, action_dim=None, chunk=64):
        """Recompute the STORE cache of every stored episode with `embed_transitions` (RNN_head.
        encode_transition_embeddings), exactly as EpisodeTrajectory.commit does at rollout time. Called after
        resuming from a buffer checkpoint, which does not contain the cache. Returns the number of episodes."""
        if self.cached_embeddings is None:
            return 0
        episodes = torch.nonzero(self.valid_index > 0).flatten()
        rows = torch.arange(self.sampled_seq_len, device=self.actions.device)
        for start in range(0, episodes.numel(), chunk):
            idx = episodes[start:start + chunk]
            row_t = rows.unsqueeze(1).expand(-1, idx.numel())
            obs, obs2 = self._observation_store.sample(idx.unsqueeze(0).expand_as(row_t), row_t)
            actions = self.actions[:, idx, :]
            if not self.act_continuous:  # stored as indices; commit embeds the one-hot actions
                actions = F.one_hot(actions.squeeze(-1).long(), num_classes=action_dim).float()
            z = embed_transitions(actions[1:], self.rewards[1:, idx, :], obs[1:], obs2[1:])
            self.cached_embeddings[:, idx, :] = torch.cat((torch.zeros_like(z[:1]), z), dim=0)
            if self.cache_step is not None:
                self.cache_step[:, idx] = self._cache_update_step
        self.cache_needs_rebuild = False
        return int(episodes.numel())

    def close(self):
        self._observation_store.close()