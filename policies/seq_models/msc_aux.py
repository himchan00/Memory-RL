"""Sub-multiset contrastive (MSC) auxiliary loss for MATE.

Method reference: mate_context_extraction.md / msc_methodology.md.

Two views of the same episode's memory are built from sub-multisets of its
transition embeddings; InfoNCE pulls same-episode views together and pushes
views of other episodes apart (one episode = one context draw). Because MATE's
memory is a running mean, any sub-multiset mean is available in O(1) from the
already-computed `cumsum` / `t_expanded` tensors — no extra forward pass.

View families (`view`), with zc_i = z_i·w_i, prior seed Ψ, prior weight W:
- "subset":   full-prefix mean  m_t = (Σ_{i≤t} zc_i + Ψ) / (t + W)  vs the
              Bernoulli(β) sub-multiset mean  (Σ b_i zc_i + βΨ) / (β(t + W)).
              The seed is subsampled by β too, keeping the view unbiased for
              m_t (Prop 2). The views share the kept transitions → per-step
              nuisance can leak into the positive pair (leakage ~ β).
- "split":    disjoint halves from mask b / (1-b); both are unbiased estimates
              of m_t that share ZERO transitions (Prop 4 exact — only the
              context links the pair). beta = split ratio (0.5 = even halves).
- "temporal": prefix [0,t] vs suffix (t,L] — disjoint AND cross-time: the
              early, few-sample memory is pulled toward what the rest of the
              episode reveals, directly training within-episode inference
              speed. Matching accuracy is floored by the Bayes error of
              context inference, so the loss does not saturate. beta unused.
split/temporal use symmetric InfoNCE (A→B and B→A).

Trainable parts:
- `log_gains`: spectral gains s = exp(log_gains) — the policy consumes s ⊙ m_t
  (applied by `Mate`), which by linearity equals the running mean of s ⊙ z: a
  diagonal reweighting of the kernel's spectral measure (Prop 5), so the
  memory stays a kernel mean embedding under the reweighted kernel.
- `head`: throwaway projection head; never used by the policy. Gives the
  InfoNCE an over-compression buffer so the memory itself is not collapsed
  onto pure context information.

With detach_inputs=True, aux gradients reach only gains + head (backward is
T-independent); the encoder is shaped by this loss only when detach_inputs=False.
The RL gradient always reaches the gains through the policy path.

`Mate` owns the three hook points (construction, forward, rollout gains), each
guarded by `self.msc is not None`; the loss reaches the RL update via the
`info["_aux_loss"]` channel. msc_enable=False reproduces the baseline exactly.
NOTE: the aux loss shares one backward and one global grad clip with the RL
loss — keep msc_lambda small (≤ 0.1) or the clipped RL gradient gets crushed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCAux(nn.Module):
    def __init__(
        self,
        hidden_size,
        msc_lambda=0.1,
        beta=0.7,
        tau=0.1,
        n_anchors=4,
        proj_dim=128,
        min_anchor_frac=0.1,
        detach_inputs=True,
        view="subset",
        focal_gamma=0.0,
        anchor_power=1.0,
    ):
        super().__init__()
        assert 0.0 < beta < 1.0, "msc beta (keep-prob / split ratio) must be in (0, 1)"
        assert tau > 0.0
        assert view in ("subset", "split", "temporal"), \
            f"msc view must be 'subset', 'split' or 'temporal' (got {view!r})"
        assert focal_gamma >= 0.0
        assert anchor_power > 0.0
        self.lam = float(msc_lambda)
        self.beta = float(beta)          # subset: keep-prob; split: ratio; temporal: unused
        self.tau = float(tau)
        self.n_anchors = int(n_anchors)
        self.min_anchor_frac = float(min_anchor_frac)
        self.detach_inputs = bool(detach_inputs)
        self.view = view
        # Focal weighting (Lin+ 2017), normalized: easy pairs (p→1) self-erase,
        # keeping gradient on hard pairs even at high accuracy. 0 = plain mean CE.
        self.focal_gamma = float(focal_gamma)
        # Anchor sampling t = lo + u^power·(hi-lo): power>1 oversamples EARLY
        # timesteps, where the running mean is noisiest. 1 = uniform.
        self.anchor_power = float(anchor_power)

        # init 0 → s = 1: the policy path starts exactly at the baseline memory.
        self.log_gains = nn.Parameter(torch.zeros(hidden_size))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def gains(self):
        return self.log_gains.exp()

    def forward(self, z_contrib, init_hidden, cumsum, t_expanded, mask=None):
        """
        z_contrib:   (T, B, D) per-step contributions z·w (post transition-dropout)
        init_hidden: (1, B, D) cumsum seed (init_emb contribution or zeros)
        cumsum:      (T, B, D) seed + running sum of z_contrib
        t_expanded:  (T, B, 1) init weight + running gated count
        mask:        (T, B, 1) optional validity mask. Padding sits at the tail,
                     so every i ≤ t is valid whenever anchor step t is.
        Returns (lambda-weighted scalar loss, info dict of detached GPU tensors).
        """
        T, B, D = z_contrib.shape
        A = self.n_anchors
        if self.detach_inputs:
            z_contrib, init_hidden = z_contrib.detach(), init_hidden.detach()
            cumsum, t_expanded = cumsum.detach(), t_expanded.detach()

        # Bernoulli-masked cumsum for subset/split, via the same cat-cumsum
        # pattern as Mate.forward (avoids pytorch/pytorch#180221). The seed is
        # scaled by beta so the sub-multiset mean stays unbiased for m_t.
        if self.view != "temporal":
            b = torch.bernoulli(z_contrib.new_full((T, B, 1), self.beta))
            csum_b = torch.cat([self.beta * init_hidden, b * z_contrib], dim=0).cumsum(dim=0)[1:]

        # A anchor timesteps per episode, t = lo + u^power·(hi-lo) in [frac·L, L-1]
        # (temporal: [frac·L, L-2], keeping ≥1 transition in the suffix).
        # power=1 → uniform; power>1 → early-t oversampling (hard anchors, where
        # the running mean is noisiest and inference speed matters).
        if mask is not None:
            lengths = mask.sum(dim=0).squeeze(-1)  # (B,)
        else:
            lengths = z_contrib.new_full((B,), float(T))
        lo = self.min_anchor_frac * lengths
        hi = lengths - (2.0 if self.view == "temporal" else 1.0)
        u = torch.rand((A, B), device=z_contrib.device)
        if self.anchor_power != 1.0:
            u = u.pow(self.anchor_power)
        t_idx = (lo + u * (hi - lo).clamp(min=0.0)).long().clamp_(min=0, max=T - 1)  # (A, B)

        idx_d = t_idx.unsqueeze(-1).expand(A, B, D)
        cnt = torch.gather(t_expanded, 0, t_idx.unsqueeze(-1)).clamp(min=1e-6)  # (A, B, 1)
        csum_t = torch.gather(cumsum, 0, idx_d)                                 # (A, B, D)

        if self.view == "temporal":
            # "The memory after t steps should already point where the rest of
            # the episode points." The suffix subtraction cancels the init seed
            # exactly, leaving a pure mean over transitions (t, L].
            end_idx = (lengths - 1.0).long().clamp(min=0, max=T - 1)            # (B,) last valid step
            end_1 = end_idx.view(1, B, 1).expand(A, B, 1)
            csum_end = torch.gather(cumsum, 0, end_idx.view(1, B, 1).expand(A, B, D))
            cnt_end = torch.gather(t_expanded, 0, end_1)
            view_a = csum_t / cnt
            view_b = (csum_end - csum_t) / (cnt_end - cnt).clamp(min=1e-6)
        elif self.view == "split":
            # view_a from mask b, view_b from mask (1-b): both unbiased for m_t
            # (pseudo-count split beta / 1-beta), sharing no transitions.
            csum_bt = torch.gather(csum_b, 0, idx_d)
            view_a = csum_bt / (self.beta * cnt)
            view_b = (csum_t - csum_bt) / ((1.0 - self.beta) * cnt)
        else:  # "subset": clean full-prefix anchor vs beta-subsampled positive
            csum_bt = torch.gather(csum_b, 0, idx_d)
            view_a = csum_t / cnt
            view_b = csum_bt / (self.beta * cnt)

        # InfoNCE on cosine similarity. Rows/cols from the same episode are
        # removed from the negatives (they share the context by construction).
        s = self.gains()
        uq = F.normalize(self.head(s * view_a).flatten(0, 1), dim=-1)  # (N, p), N = A·B
        vk = F.normalize(self.head(s * view_b).flatten(0, 1), dim=-1)  # (N, p)
        logits = uq @ vk.t() / self.tau                                # (N, N)
        ep = torch.arange(B, device=logits.device).repeat(A)           # episode id per row
        same_ep = ep.unsqueeze(0) == ep.unsqueeze(1)
        diag = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_ep & ~diag, float("-inf"))
        target = torch.arange(logits.shape[0], device=logits.device)
        ce = F.cross_entropy(logits, target, reduction="none")         # (N,)
        if self.view in ("split", "temporal"):
            # Both views are noisy → symmetrize. The same-episode mask is
            # symmetric, so the masked B→A logits are exactly the transpose.
            ce = 0.5 * (ce + F.cross_entropy(logits.t(), target, reduction="none"))

        if self.focal_gamma > 0.0:
            # Normalized focal weighting: easy pairs (p→1) self-erase; the loss
            # is the weighted mean CE over the pairs that are still hard. The
            # normalization keeps the loss scale bounded, but the gradient can
            # still concentrate — watch raw_grad_norm when enabling this.
            p = ce.detach().neg().exp()                # per-pair prob of correct match
            w = (1.0 - p).pow(self.focal_gamma)        # (N,), detached weights
            loss = (w * ce).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = ce.mean()

        info = {
            "msc_loss": loss.detach(),
            "msc_pos_sim": (uq * vk).sum(-1).detach().mean(),
            "msc_acc": (logits.detach().argmax(dim=1) == target).float().mean(),
            "msc_gain_mean": s.detach().mean(),
            "msc_gain_std": s.detach().std(),
        }
        if self.focal_gamma > 0.0:
            info["msc_hard_frac"] = w.mean()  # effective fraction of pairs still training
        return self.lam * loss, info
