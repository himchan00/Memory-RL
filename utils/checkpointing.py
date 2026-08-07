import os
import pickle
import time
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2
_RETRIABLE_SAVE_ERRORS = (OSError, RuntimeError, EOFError, pickle.UnpicklingError)


def load_training_checkpoint(path: str, map_location: Any = "cpu") -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format {version!r}; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}. "
            "Checkpoints created before the core refactor are not supported."
        )
    return checkpoint


def save_verified(
    obj: Any,
    final_path: str,
    *,
    attempts: int = 3,
) -> bool:
    """Write, reload, and atomically replace a checkpoint with bounded retries."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    tmp_path = final_path + ".tmp"
    for attempt in range(attempts):
        try:
            torch.save(obj, tmp_path)
            torch.load(tmp_path, map_location="cpu", weights_only=False)
            os.replace(tmp_path, final_path)
            return True
        except _RETRIABLE_SAVE_ERRORS as exc:
            _remove_if_present(tmp_path)
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
                print(
                    f"[WARN] checkpoint save attempt {attempt + 1}/{attempts} "
                    f"failed ({type(exc).__name__}: {exc}); retrying"
                )
            else:
                print(
                    f"[WARN] checkpoint save failed after {attempts} attempts; "
                    f"deferring to the next save interval ({type(exc).__name__}: {exc})"
                )
                return False
    return False


def _remove_if_present(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[WARN] failed to remove temporary checkpoint {path}: {exc}")

