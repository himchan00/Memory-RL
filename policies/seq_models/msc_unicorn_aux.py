"""UNICORN loss for MATE — usable in joint, alternating_ema and alternating_online.

Reference: "Towards an Information Theoretic Framework of Context-Based
Offline Meta-Reinforcement Learning" (NeurIPS 2024, arXiv:2402.02429),
self-supervised variant (UNICORN-SS):

    L = L_recon + (alpha / (1 - alpha)) * L_FOCAL,      alpha in (0, 1)

grounded in the decomposition  I(Z;X_t|X_b) <= I(Z;M) <= I(Z;X):
- L_FOCAL  approximates the UPPER bound I(Z;X) — distance metric learning
  that pulls same-task representations together and pushes different tasks
  apart (FOCAL, Li et al. 2021).
- L_recon  approximates the LOWER bound I(Z;X_t|X_b) — a decoder predicts
  the task-related component x_t = (r, s'-part) from (Z, x_b = (s, a)).
- alpha interpolates between the bounds (paper: robust over a wide range;
  their table uses alpha/(1-alpha) in ~0.15-1.5).

Adaptation to MATE's episode-as-task setting:
- Task identity = episode identity (one context draw per episode).
- FOCAL pair: the memories of two DISJOINT random half-subsets of the same
  episode (MATE mean with the detached prior seed), so a positive pair
  shares only the context; negatives are cross-episode pairs. The metric
  acts directly on memory space (no projection head), as in FOCAL where z
  is the policy input.
- Recon: leave-one-out memory Z_-j = (S - z_j)/(N - 1) per valid step j,
  decoder([Z_-j, x_b_j]) -> x_t_j, masked MSE. j must be excluded or the
  response sits inside the encoder's own input (identity shortcut).

Mode support (the reason this module exists as one callable):
- joint: Mate.forward calls it with apply_lambda=True; msc_lambda scales the
  loss into the shared RL backward (keep lambda <= 0.1).
- alternating_ema / alternating_online: Mate.contrastive_loss calls it with
  apply_lambda=False; the separate msc optimizer owns encoder(+decoder).

With detach_z=True only the decoder trains (FOCAL has no parameters), so the
encoder is not shaped at all — use detach_z=False for any run where UNICORN
is meant to do representation learning.
"""
import torch
import torch.nn as nn


class MSCUnicornAux(nn.Module):
    _FOCAL_EPS = 0.1  # FOCAL negative-pair epsilon: beta / (dist^2 + eps)

    def __init__(
        self,
        hidden_size,
        query_dim,
        response_dim,
        msc_lambda=0.05,
        alpha=0.5,
        recon_hidden=256,
        detach_z=False,
    ):
        super().__init__()
        assert 0.0 < alpha < 1.0, "unicorn alpha must be in (0, 1)"
        assert query_dim and query_dim > 0 and response_dim > 0
        self.lam = float(msc_lambda)
        self.alpha = float(alpha)
        self.query_dim = int(query_dim)
        self.detach_z = bool(detach_z)

        # Throwaway decoder for the lower-bound term; never on the policy path.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + self.query_dim, int(recon_hidden)),
            nn.ReLU(),
            nn.Linear(int(recon_hidden), int(response_dim)),
        )

    def forward(
        self,
        inputs,
        z,
        init_hidden,
        init_count,
        cumsum,
        mask=None,
        *,
        apply_lambda=True,
    ):
        """
        inputs:      (T, B, input_size) raw transition tuples [query | response]
        z:           (T, B, D) transition embeddings (grad -> encoder unless detach_z)
        init_hidden: (1, B, D) prior seed; init_count: (1, B, 1) prior count
        cumsum:      (T, B, D) seed + running sum of z
        mask:        (T, B, 1) validity mask (padding at the tail)
        Returns (loss, info dict of detached GPU tensors).
        """
        T, B, D = z.shape
        if B < 2:
            raise ValueError("UNICORN needs at least two episodes per batch")
        if self.detach_z:
            z, cumsum = z.detach(), cumsum.detach()
        if mask is None:
            valid = torch.ones((T, B), dtype=torch.bool, device=z.device)
        else:
            valid = mask.squeeze(-1).bool()
        lengths = valid.sum(dim=0)  # (B,) long

        # ---- FOCAL term (upper bound): disjoint half-subset memories --------
        order = torch.rand((T, B), device=z.device).masked_fill(
            ~valid, float("inf")
        ).argsort(dim=0)
        rank = torch.empty_like(order)
        rank.scatter_(
            0, order,
            torch.arange(T, device=z.device).unsqueeze(1).expand(T, B),
        )
        half = lengths.div(2, rounding_mode="floor").clamp(min=1)
        m_a = (rank < half.unsqueeze(0)).to(z.dtype).unsqueeze(-1)
        m_b = ((rank >= half.unsqueeze(0)) & (rank < (2 * half).unsqueeze(0))
               ).to(z.dtype).unsqueeze(-1)
        seed = init_hidden.detach().squeeze(0)
        seed_n = init_count.detach().squeeze(0)

        def mate_mean(m):
            return (seed + (z * m).sum(dim=0)) / (
                seed_n + m.sum(dim=0)
            ).clamp_min(1e-6)

        mem_a, mem_b = mate_mean(m_a), mate_mean(m_b)          # (B, D)
        d2 = torch.cdist(mem_a, mem_b).pow(2)                  # (B, B)
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        pos = d2[eye].mean()
        neg = (1.0 / (d2[~eye] + self._FOCAL_EPS)).mean()
        focal = pos + neg

        # ---- Recon term (lower bound): LOO memory -> response ---------------
        end = (lengths - 1.0).long().clamp(min=0, max=T - 1)
        total_z = torch.gather(cumsum, 0, end.view(1, B, 1).expand(1, B, D))
        total_n = init_count.detach() + lengths.view(1, B, 1)
        z_loo = (total_z - z) / (total_n - 1.0).clamp(min=1e-6)  # (T, B, D)
        query = inputs[..., : self.query_dim]
        target = inputs[..., self.query_dim:]
        pred = self.decoder(torch.cat([z_loo, query], dim=-1))
        se = (pred - target).pow(2).mean(dim=-1, keepdim=True)
        if mask is None:
            recon = se.mean()
        else:
            recon = (se * mask).sum() / mask.sum().clamp(min=1.0)

        loss = recon + (self.alpha / (1.0 - self.alpha)) * focal
        info = {
            "msc_unicorn_recon": recon.detach(),
            "msc_unicorn_focal": focal.detach(),
            "msc_focal_pos_d2": pos.detach(),
            "msc_focal_neg_d2": d2[~eye].detach().mean(),
            "msc_loss": loss.detach(),
        }
        return (self.lam * loss if apply_lambda else loss), info
