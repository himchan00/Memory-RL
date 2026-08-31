from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RecurrentBatch:
    actions: torch.Tensor
    rewards: torch.Tensor
    observs: torch.Tensor
    next_observs: torch.Tensor
    terms: torch.Tensor
    masks: torch.Tensor
    transition_t: torch.Tensor


def prepare_recurrent_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    discrete_action_dim: int | None = None,
) -> RecurrentBatch:
    actions = batch["act"]
    if discrete_action_dim is not None:
        actions = F.one_hot(
            actions.squeeze(-1).long(),
            num_classes=discrete_action_dim,
        ).float()

    return RecurrentBatch(
        actions=actions,
        rewards=batch["rew"],
        observs=batch["obs"],
        next_observs=batch["obs2"],
        terms=batch["term"],
        masks=batch["mask"],
        transition_t=batch["transition_t"],
    )


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> dict[str, torch.Tensor | float]:
    params = tuple(parameters)
    grad_norm = nn.utils.clip_grad_norm_(params, max_norm)
    return {
        "raw_grad_norm": grad_norm.detach(),
        "grad_clip_coef": torch.clamp(
            max_norm / (grad_norm.detach() + 1e-12),
            max=1.0,
        ),
        "clip_grad_norm": max_norm,
    }
