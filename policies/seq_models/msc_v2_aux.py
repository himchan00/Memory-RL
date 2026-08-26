import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCV2Aux(nn.Module):
    def __init__(
        self,
        hidden_size,
        msc_lambda=0.05,
        tau=0.1,
        k_min=8,
        k_max=64,
        detach_z=False,
    ):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(float(msc_lambda)) or msc_lambda < 0.0:
            raise ValueError("msc_lambda must be finite and non-negative")
        if not math.isfinite(float(tau)) or tau <= 0.0:
            raise ValueError("msc_tau must be finite and positive")
        if (
            isinstance(k_min, bool)
            or not isinstance(k_min, int)
            or isinstance(k_max, bool)
            or not isinstance(k_max, int)
            or k_min <= 0
            or k_max < k_min
        ):
            raise ValueError("msc_k_min and msc_k_max must satisfy 0 < k_min <= k_max")

        self.lam = float(msc_lambda)
        self.k_min = k_min
        self.k_max = k_max
        self.detach_z = bool(detach_z)
        self.weight = nn.Parameter(torch.eye(hidden_size))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(float(tau))))

    def forward(
        self,
        z,
        init_hidden,
        init_count,
        gate_weights,
        mask=None,
        *,
        apply_lambda=True,
    ):
        if z.dim() != 3:
            raise ValueError("z must have shape (T, B, D)")
        time, batch_size, hidden_size = z.shape
        if batch_size < 2:
            raise ValueError("MSC v2 requires at least two episodes per batch")
        if hidden_size != self.weight.shape[0]:
            raise ValueError(
                f"z has hidden size {hidden_size}, expected {self.weight.shape[0]}"
            )
        if init_hidden.shape != (1, batch_size, hidden_size):
            raise ValueError("init_hidden must have shape (1, B, D)")
        if init_count.shape != (1, batch_size, 1):
            raise ValueError("init_count must have shape (1, B, 1)")
        if gate_weights.shape != (time, batch_size, 1):
            raise ValueError("gate_weights must have shape (T, B, 1)")

        if mask is None:
            valid = torch.ones(
                (time, batch_size),
                dtype=torch.bool,
                device=z.device,
            )
        else:
            if mask.shape != (time, batch_size, 1):
                raise ValueError("mask must have shape (T, B, 1)")
            valid = mask.squeeze(-1).bool()

        lengths = valid.sum(dim=0)
        min_length = lengths.min()
        capacity = min_length.div(2, rounding_mode="floor")
        k_lower = torch.minimum(
            capacity,
            min_length.new_tensor(self.k_min),
        )
        k_upper = torch.minimum(
            capacity,
            min_length.new_tensor(self.k_max),
        )
        k = k_lower + torch.floor(
            torch.rand((), device=z.device) * (k_upper - k_lower + 1)
        ).long()
        has_split = (capacity > 0).to(z.dtype)

        random_keys = torch.rand((time, batch_size), device=z.device)
        order = random_keys.masked_fill(~valid, float("inf")).argsort(dim=0)
        ordered_z = torch.gather(
            z.detach() if self.detach_z else z,
            0,
            order.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        ordered_weights = torch.gather(
            gate_weights.detach().squeeze(-1),
            0,
            order,
        )

        rank = torch.arange(time, device=z.device).unsqueeze(1)
        mask_a = (rank < k).to(dtype=z.dtype)
        mask_b = ((rank >= k) & (rank < 2 * k)).to(dtype=z.dtype)
        init_hidden = init_hidden.detach().squeeze(0)
        init_count = init_count.detach().squeeze(0)

        def subset_mean(subset_mask):
            weights = ordered_weights * subset_mask
            numerator = init_hidden + (
                ordered_z * weights.unsqueeze(-1)
            ).sum(dim=0)
            denominator = init_count + weights.sum(dim=0, keepdim=False).unsqueeze(-1)
            return numerator / denominator.clamp_min(1e-6)

        memory_a = subset_mean(mask_a)
        memory_b = subset_mean(mask_b)
        kappa = self.log_kappa.exp().clamp_min(1e-6)
        logits_fwd = (memory_a @ self.weight.t()) @ memory_b.t() / kappa
        logits_bwd = (memory_b @ self.weight.t()) @ memory_a.t() / kappa
        targets = torch.arange(batch_size, device=z.device)
        loss_fwd = has_split * F.cross_entropy(logits_fwd, targets)
        loss_bwd = has_split * F.cross_entropy(logits_bwd, targets)
        loss = 0.5 * (loss_fwd + loss_bwd)

        acc_fwd = has_split * (
            logits_fwd.detach().argmax(dim=1) == targets
        ).float().mean()
        acc_bwd = has_split * (
            logits_bwd.detach().argmax(dim=1) == targets
        ).float().mean()
        info = {
            "msc_loss": loss.detach(),
            "msc_loss_fwd": loss_fwd.detach(),
            "msc_loss_bwd": loss_bwd.detach(),
            "msc_acc": 0.5 * (acc_fwd + acc_bwd),
            "msc_acc_fwd": acc_fwd,
            "msc_acc_bwd": acc_bwd,
            "msc_mi_lower_bound": has_split * (
                loss.new_tensor(math.log(batch_size)) - loss.detach()
            ),
            "msc_k": k.detach().to(loss.dtype),
            "msc_memory_a_norm": has_split * memory_a.detach().norm(dim=-1).mean(),
            "msc_memory_b_norm": has_split * memory_b.detach().norm(dim=-1).mean(),
            "msc_kappa": kappa.detach(),
            "msc_skipped": 1.0 - has_split,
        }
        return (self.lam * loss if apply_lambda else loss), info
