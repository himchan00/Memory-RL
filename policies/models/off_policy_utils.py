import copy
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

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
    cached_embeddings: torch.Tensor | None
    cached_prefixes: torch.Tensor | None
    # STORE independent loss rows: (act, rew, obs, obs2, mask, transition_t,
    # cached_embeddings) at the rows whose embeddings are recomputed.
    store_rows: tuple | None = None
    # STORE subset: expected valid loss rows (k/T of the full-episode count).
    num_valid: torch.Tensor | None = None


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

    store_rows = None
    if "store" in batch:
        store = prepare_recurrent_batch(
            batch["store"], discrete_action_dim=discrete_action_dim
        )
        store_rows = (
            store.actions, store.rewards, store.observs, store.next_observs,
            store.masks, store.transition_t, store.cached_embeddings,
        )

    return RecurrentBatch(
        actions=actions,
        rewards=batch["rew"],
        observs=batch["obs"],
        next_observs=batch["obs2"],
        terms=batch["term"],
        masks=batch["mask"],
        transition_t=batch["transition_t"],
        cached_embeddings=batch.get("cached_embeddings"),
        cached_prefixes=batch.get("cached_prefixes"),
        store_rows=store_rows,
        num_valid=batch.get("num_valid"),
    )


def module_grad_norms(
    groups: Mapping[str, Iterable[nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """L2 gradient norm of each named parameter group as a 0-dim GPU tensor
    (no host sync). Logged as `grad_norm/<group>` every update so a module that
    silently stops receiving gradient is visible in W&B, and asserted on at log
    time by the Learner."""
    norms = {}
    for name, params in groups.items():
        params = tuple(params)
        grads = [p.grad for p in params if p.grad is not None]
        if grads:
            # one fused per-tensor norm kernel per group (as clip_grad_norm_ does)
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))
        else:
            norm = params[0].new_zeros(())
        norms[f"grad_norm/{name}"] = norm.detach()
    return norms


def compare_eager_compiled_gradients(
    module: nn.Module,
    run_backward: Callable[[Callable], None],
    eager_fn: Callable,
    compiled_fn: Callable,
    groups: Mapping[str, Iterable[nn.Parameter]],
    *,
    ratio_bounds: tuple[float, float] = (0.25, 4.0),
) -> dict[str, dict[str, float]]:
    """Backprop one batch through the eager loss and through the compiled loss
    that training will use, and compare the per-group gradient norms.

    Guards against a miscompiled backward: Inductor (torch <= 2.9) turned the
    gradient of MATE's STORE path into an exact zero while every other module
    trained normally, which no loss curve reveals. Norms rather than vectors are
    compared because eager and Inductor draw different dropout masks. Module
    state (PopArt / input-norm statistics, EMA buffers) is restored afterwards
    and no optimizer step is taken; `run_backward(fn)` must evaluate the loss
    with `fn` and call `.backward()`.
    """
    snapshot = copy.deepcopy(module.state_dict())
    norms = {}
    try:
        for tag, fn in (("eager", eager_fn), ("compiled", compiled_fn)):
            module.zero_grad(set_to_none=True)
            run_backward(fn)
            norms[tag] = {
                key: float(value) for key, value in module_grad_norms(groups).items()
            }
            module.load_state_dict(snapshot)
    finally:
        module.zero_grad(set_to_none=True)

    low, high = ratio_bounds
    rows, dead = [], []
    for key, eager in norms["eager"].items():
        compiled = norms["compiled"][key]
        ok = (eager == 0.0 and compiled == 0.0) or (
            eager > 0.0 and compiled > 0.0 and low <= compiled / eager <= high
        )
        rows.append(f"  {key:26s} eager={eager:.3e}  compiled={compiled:.3e}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            dead.append(key)
    print("[compile self-test] per-module gradient norms on one batch:\n" + "\n".join(rows))
    if dead:
        raise RuntimeError(
            "torch.compile produced a gradient that disagrees with eager mode for "
            f"{dead}; the compiled training graph is wrong (a module would silently "
            "stop learning). Run with --config_seq.compile=False or fix the graph; "
            "see CLAUDE.md 'Gradient guards'."
        )
    return norms


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
