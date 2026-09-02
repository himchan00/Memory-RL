"""Scale-covering episode CPC (v3) for MATE's representation step.

Reference: the "Scale-Covering Episode CPC" note (supersedes the
query-conditioned MSC note). Environment-invariant by construction (R1) and
reducing to plain bilinear CPC when its additions are unneeded (R2):

- **Scale covering**: the two disjoint subset sizes k_u, k_v are drawn
  PER EPISODE from LogUniform[1, floor(L/2)]. The Bayes discriminability
  curve D(k) is monotone in k, so whatever the environment or training
  stage, some sampled k lands where the InfoNCE gradient is alive — no
  k_min/k_max tuning, and transition (k=1) / split (k=L/2) views become the
  endpoints of one distribution.
- **Candidate-side quadratic**: heterogeneous per-episode k makes candidate
  noise levels differ, so the CLT-optimal critic is
  s(u, v) = uᵀWv − ½·vᵀAv (the quadratic no longer cancels in the softmax).
  A is initialized to zero, so the critic STARTS as exactly the bilinear
  one; the term only acts if heterogeneity makes it useful (R2).
- **Query-twin diagnostic (always on)**: a small twin encoder sees only the
  query slice x_b = (s, a) of each transition and solves the same CPC over
  the same subsets with its own critic. Its accuracy `msc_accq` IS the
  environment's shortcut amount, measured instead of analyzed: high accq
  means episode identity is readable from queries alone (memory need not
  carry it). The twin trains on its own loss; no gradient path touches the
  main encoder or critic, so the diagnostic cannot change the learned
  representation. `twin_correction=True` (off by default; gate on accq)
  subtracts the DETACHED twin logits from the main logits, scoring only the
  information the twin cannot already explain.

Interface mirrors MSCV2Aux and is consumed through `Mate.contrastive_loss`
(alternating_ema only). Symmetric InfoNCE (both directions), learnable
temperature kappa as in v2.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCV3Aux(nn.Module):
    def __init__(
        self,
        hidden_size,
        query_dim,
        msc_lambda=0.05,
        tau=0.1,
        twin_hidden=64,
        twin_correction=False,
        detach_z=False,
    ):
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if not math.isfinite(float(tau)) or tau <= 0.0:
            raise ValueError("msc_tau must be finite and positive")
        if query_dim is None or not 0 < int(query_dim):
            raise ValueError("v3 requires a positive msc_query_dim (for the query twin)")
        self.lam = float(msc_lambda)
        self.query_dim = int(query_dim)
        self.detach_z = bool(detach_z)
        self.twin_correction = bool(twin_correction)

        # Main critic: W init identity (= v2's bilinear start), A init zero
        # (quadratic off at init -> exact v2-shaped critic until learned).
        self.weight = nn.Parameter(torch.eye(hidden_size))
        self.quad = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        self.log_kappa = nn.Parameter(torch.tensor(math.log(float(tau))))

        # Query twin: independent tiny encoder + critic on the query slice.
        th = int(twin_hidden)
        self.twin_encoder = nn.Sequential(
            nn.Linear(self.query_dim, th), nn.ReLU(), nn.Linear(th, th),
        )
        self.twin_weight = nn.Parameter(torch.eye(th))
        self.twin_quad = nn.Parameter(torch.zeros(th, th))
        self.twin_log_kappa = nn.Parameter(torch.tensor(math.log(float(tau))))

    @staticmethod
    def _log_uniform_k(cap, shape, device):
        """Integer k ~ LogUniform[1, cap], elementwise (cap: (B,) long, >= 1)."""
        u = torch.rand(shape, device=device)
        hi = (cap.to(torch.float32) + 1.0).log()
        k = torch.exp(u * hi).floor().long()
        return k.clamp(min=1).minimum(cap)

    def _cpc(self, mem_a, mem_b, weight, quad, log_kappa):
        """Symmetric InfoNCE with candidate-side quadratic. mem_*: (B, D)."""
        kappa = log_kappa.exp().clamp_min(1e-6)
        targets = torch.arange(mem_a.shape[0], device=mem_a.device)

        def one_way(anchor, cand):
            bilinear = (anchor @ weight) @ cand.t()               # (B, B)
            quad_pen = 0.5 * ((cand @ quad) * cand).sum(-1)       # (B,) per candidate
            return (bilinear - quad_pen.unsqueeze(0)) / kappa

        logits_fwd = one_way(mem_a, mem_b)
        logits_bwd = one_way(mem_b, mem_a)
        return logits_fwd, logits_bwd, targets

    def forward(
        self,
        inputs,
        z,
        init_hidden,
        init_count,
        mask=None,
        *,
        apply_lambda=True,
    ):
        """
        inputs:      (T, B, input_size) raw transition tuples (twin query source)
        z:           (T, B, D) online-encoder embeddings
        init_hidden: (1, B, D) prior seed; init_count: (1, B, 1) prior count
        mask:        (T, B, 1) optional validity mask (padding at the tail)
        """
        time, batch_size, hidden_size = z.shape
        if batch_size < 2:
            raise ValueError("MSC v3 requires at least two episodes per batch")

        if mask is None:
            valid = torch.ones((time, batch_size), dtype=torch.bool, device=z.device)
        else:
            valid = mask.squeeze(-1).bool()
        lengths = valid.sum(dim=0)                                 # (B,) long
        capacity = lengths.div(2, rounding_mode="floor")
        # Mirrors v2's guard: a batch containing an unsplittable episode
        # contributes a zeroed loss instead of crashing under torch.compile.
        has_split = (capacity.min() > 0).to(z.dtype)
        cap = capacity.clamp(min=1)

        # Per-episode scale covering: k_u, k_v ~ LogU[1, cap_b], independent.
        k_u = self._log_uniform_k(cap, (batch_size,), z.device)    # (B,)
        k_v = self._log_uniform_k(cap, (batch_size,), z.device)
        # k_u + k_v <= 2*cap <= L, so disjointness always fits.

        # Disjoint subsets via a per-episode random order over valid steps:
        # ranks [0, k_u) -> U, ranks [k_u, k_u+k_v) -> V.
        order = torch.rand((time, batch_size), device=z.device).masked_fill(
            ~valid, float("inf")
        ).argsort(dim=0)
        rank = torch.empty_like(order)
        rank.scatter_(
            0, order,
            torch.arange(time, device=z.device).unsqueeze(1).expand(time, batch_size),
        )                                                          # rank[t,b] of step t
        mask_u = (rank < k_u.unsqueeze(0)).to(z.dtype).unsqueeze(-1)          # (T,B,1)
        mask_v = ((rank >= k_u.unsqueeze(0)) & (rank < (k_u + k_v).unsqueeze(0))
                  ).to(z.dtype).unsqueeze(-1)

        z_used = z.detach() if self.detach_z else z
        seed = init_hidden.detach().squeeze(0)                     # (B, D)
        seed_n = init_count.detach().squeeze(0)                    # (B, 1)

        def mate_mean(x, m, use_prior):
            s = (x * m).sum(dim=0)
            n = m.sum(dim=0)
            if use_prior:
                s = s + seed
                n = n + seed_n
            return s / n.clamp_min(1e-6)

        mem_u = mate_mean(z_used, mask_u, True)
        mem_v = mate_mean(z_used, mask_v, True)

        # Query twin on the same subsets (detached input slice; its own params).
        zq = self.twin_encoder(inputs[..., : self.query_dim].detach())
        twin_u = mate_mean(zq, mask_u, False)
        twin_v = mate_mean(zq, mask_v, False)
        tl_fwd, tl_bwd, targets = self._cpc(
            twin_u, twin_v, self.twin_weight, self.twin_quad, self.twin_log_kappa
        )
        twin_loss = 0.5 * (
            F.cross_entropy(tl_fwd, targets) + F.cross_entropy(tl_bwd, targets)
        )

        logits_fwd, logits_bwd, _ = self._cpc(
            mem_u, mem_v, self.weight, self.quad, self.log_kappa
        )
        if self.twin_correction:
            # Gated shortcut correction: score only what the twin cannot
            # already explain. Detached, so the twin stays a pure measurer.
            logits_fwd = logits_fwd - tl_fwd.detach()
            logits_bwd = logits_bwd - tl_bwd.detach()
        loss_main = 0.5 * (
            F.cross_entropy(logits_fwd, targets) + F.cross_entropy(logits_bwd, targets)
        )
        # Twin loss trains only twin parameters (its input is detached); added
        # here so the single msc optimizer covers it.
        loss = has_split * (loss_main + twin_loss)

        acc = 0.5 * (
            (logits_fwd.detach().argmax(1) == targets).float().mean()
            + (logits_bwd.detach().argmax(1) == targets).float().mean()
        )
        accq = 0.5 * (
            (tl_fwd.detach().argmax(1) == targets).float().mean()
            + (tl_bwd.detach().argmax(1) == targets).float().mean()
        )
        info = {
            "msc_loss": loss_main.detach(),
            "msc_acc": has_split * acc,
            "msc_accq": has_split * accq,                  # shortcut amount (twin)
            "msc_twin_loss": twin_loss.detach(),
            "msc_mi_lower_bound": has_split * (
                loss_main.new_tensor(math.log(batch_size)) - loss_main.detach()
            ),
            "msc_k_u_mean": k_u.to(z.dtype).mean(),
            "msc_k_v_mean": k_v.to(z.dtype).mean(),
            "msc_kappa": self.log_kappa.detach().exp(),
            "msc_quad_norm": self.quad.detach().norm(),
            "msc_skipped": 1.0 - has_split,
        }
        return (self.lam * loss if apply_lambda else loss), info
