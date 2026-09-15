import os
from typing import Sequence

import numpy as np
import torch

from torchkit import pytorch_utils as ptu


class RamObservationStore:
    backend = "ram"

    def __init__(
        self,
        *,
        sampled_seq_len: int,
        num_episodes: int,
        observation_dim: int,
        dtype: np.dtype,
    ):
        shape = (sampled_seq_len, num_episodes, observation_dim)
        # obs_dtype="uint8" keeps pixel buffers 4x smaller; batches are cast on sample.
        self.dtype = getattr(torch, np.dtype(dtype).name)
        self.observations = ptu.zeros(shape, dtype=self.dtype)
        self.next_observations = ptu.zeros(shape, dtype=self.dtype)

    @property
    def size_gb(self) -> float:
        return (
            self.observations.numel() * self.observations.element_size() * 2
        ) / (1024**3)

    def write(
        self,
        indices: Sequence[int],
        observations: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> None:
        self.observations[:, indices, :] = self._to_storage(observations)
        self.next_observations[:, indices, :] = self._to_storage(next_observations)

    def _to_storage(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.detach()
        if not self.dtype.is_floating_point:
            observations = observations.clamp(0, torch.iinfo(self.dtype).max)
        return observations.to(self.dtype)

    def sample(self, episode_indices: torch.Tensor, row_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.observations[row_indices, episode_indices, :].float(),
            self.next_observations[row_indices, episode_indices, :].float(),
        )

    def state_dict(self) -> dict:
        return {
            "observations": self.observations.cpu(),
            "next_observations": self.next_observations.cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.observations = state_dict["observations"].to(ptu.device)
        self.next_observations = state_dict["next_observations"].to(ptu.device)

    def close(self) -> None:
        return


class MemmapObservationStore:
    backend = "memmap"

    def __init__(
        self,
        *,
        sampled_seq_len: int,
        num_episodes: int,
        observation_dim: int,
        dtype: np.dtype,
        directory: str,
    ):
        os.makedirs(directory, exist_ok=True)
        self.shape = (num_episodes, sampled_seq_len, observation_dim)
        self.dtype = dtype
        self.directory = directory
        self.obs_path = os.path.join(directory, "obs.dat")
        self.next_obs_path = os.path.join(directory, "next_obs.dat")
        self.observations = self._open(self.obs_path)
        self.next_observations = self._open(self.next_obs_path)

    def _open(self, path: str) -> np.memmap:
        # "w+" zeroes the file, which would wipe the observations a --resume is
        # about to restore; reuse an existing file of the expected size instead.
        nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        exists = os.path.exists(path) and os.path.getsize(path) == nbytes
        return np.memmap(
            path,
            dtype=self.dtype,
            mode="r+" if exists else "w+",
            shape=self.shape,
        )

    @property
    def size_gb(self) -> float:
        return (
            np.prod(self.shape) * self.dtype.itemsize * 2
        ) / (1024**3)

    def write(
        self,
        indices: Sequence[int],
        observations: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> None:
        obs_np = self._to_storage_array(observations)
        next_obs_np = self._to_storage_array(next_observations)
        self.observations[np.asarray(indices)] = obs_np
        self.next_observations[np.asarray(indices)] = next_obs_np

    def _to_storage_array(self, observations: torch.Tensor) -> np.ndarray:
        array = observations.detach().cpu().numpy()
        if np.issubdtype(self.dtype, np.integer):
            array = np.clip(array, 0, np.iinfo(self.dtype).max)
        return np.transpose(array, (1, 0, 2)).astype(
            self.dtype,
            copy=False,
        )

    def sample(self, episode_indices: torch.Tensor, row_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        episode_array = episode_indices.detach().cpu().numpy()
        row_array = row_indices.detach().cpu().numpy()
        return tuple(
            torch.from_numpy(np.ascontiguousarray(array)).to(
                ptu.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            for array in (
                self.observations[episode_array, row_array],
                self.next_observations[episode_array, row_array],
            )
        )

    def state_dict(self) -> dict:
        self.observations.flush()
        self.next_observations.flush()
        return {
            "memmap_dir": self.directory,
            "obs_dtype": str(self.dtype),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        saved_dtype = np.dtype(state_dict["obs_dtype"])
        if saved_dtype != self.dtype:
            raise ValueError(
                f"Saved observation dtype {saved_dtype} does not match {self.dtype}"
            )
        self.close()
        self.observations = np.memmap(
            self.obs_path,
            dtype=self.dtype,
            mode="r+",
            shape=self.shape,
        )
        self.next_observations = np.memmap(
            self.next_obs_path,
            dtype=self.dtype,
            mode="r+",
            shape=self.shape,
        )

    def close(self) -> None:
        if getattr(self, "observations", None) is not None:
            self.observations.flush()
            del self.observations
        if getattr(self, "next_observations", None) is not None:
            self.next_observations.flush()
            del self.next_observations
