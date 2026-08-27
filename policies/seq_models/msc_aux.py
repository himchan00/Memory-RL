"""Sub-multiset contrastive (MSC) auxiliary loss for MATE.

Method reference: mate_context_extraction.md / msc_methodology.md.

Two views of the same episode's memory are built from sub-multisets of its
transition embeddings; InfoNCE pulls same-episode views together and pushes
views of other episodes apart (one episode = one context draw). Because MATE's
memory is a running mean, any sub-multiset mean is available in O(1) from the
already-computed `cumsum` tensor and the initial count — no extra forward pass.

View families (`view`), with transition embedding z_i, prior seed Ψ, and
prior count W:
- "subset":   full-prefix mean  m_t = (Σ_{i≤t} z_i + Ψ) / (t + W)  vs the
              Bernoulli(β) sub-multiset mean  (Σ b_i z_i + βΨ) / (β(t + W)).
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
- "prefix":   two full-prefix means m_t vs m_s at independently drawn
              timesteps — the "natural" view: no masking machinery at all,
              nested and leaky (the shorter prefix is contained in the longer
              one). beta unused.
- "transition": two INDEPENDENT single transition embeddings z_i, z_j of the
              same episode — the finest granularity of the same objective. No
              memory aggregation appears in the loss at all; the running mean
              separates contexts automatically by linearity if single-z is
              context-discriminative. Accuracy is floored by the SINGLE-
              transition Bayes error (much higher than any mean view), so this
              is the most saturation-resistant variant. beta unused; requires
              detach_inputs=False (its only purpose is shaping the encoder).
split/temporal/prefix/transition use symmetric InfoNCE (A→B and B→A).

Trainable parts:
- `log_gains`: spectral gains s = exp(log_gains) — the policy consumes s ⊙ m_t
  (applied by `Mate`), which by linearity equals the running mean of s ⊙ z: a
  diagonal reweighting of the kernel's spectral measure (Prop 5), so the
  memory stays a kernel mean embedding under the reweighted kernel.
  learn_gains=False turns them into a zeros buffer (s ≡ 1 forever): the policy
  path is then an exact no-op and the aux touches only head + encoder.
- `head`: throwaway projection head; never used by the policy. Gives the
  InfoNCE an over-compression buffer so the memory itself is not collapsed
  onto pure context information.

With detach_inputs=True, aux gradients reach only gains + head (backward is
T-independent); the encoder is shaped by this loss only when detach_inputs=False
(required for view="transition"). The RL gradient always reaches the gains
through the policy path, unless learn_gains=False.

`Mate` owns the three hook points (construction, forward, rollout gains), each
guarded by `self.msc is not None`. In the default "joint" mode, the loss reaches
the RL update through `info["_aux_loss"]`; msc_lambda therefore controls its
weight in the shared backward and should remain small. In opt-in
"alternating_ema" mode, a separate optimizer minimizes raw InfoNCE over the
online transition encoder and projection head, then updates an EMA transition
encoder used by RL and rollout. Gains and all other policy-path parameters are
detached from that contrastive backward. msc_enable=False reproduces the
baseline exactly.
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
        learn_gains=True,
        pair_gap=0,
    ):
        super().__init__()
        assert 0.0 < beta < 1.0, "msc beta (keep-prob / split ratio) must be in (0, 1)"
        assert tau > 0.0
        assert view in ("subset", "split", "temporal", "prefix", "transition"), \
            f"msc view must be 'subset', 'split', 'temporal', 'prefix' or 'transition' (got {view!r})"
        assert focal_gamma >= 0.0
        assert anchor_power > 0.0
        assert pair_gap >= 0
        # The transition view's whole point is shaping the encoder: with detached
        # inputs it would train only the throwaway head (structurally inert).
        assert not (view == "transition" and detach_inputs), \
            "msc view 'transition' requires msc_detach_z=False (otherwise only the throwaway head trains)"
        self.lam = float(msc_lambda)
        self.beta = float(beta)          # subset: keep-prob; split: ratio; other views: unused
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
        # transition view only: enforce |i - j| >= pair_gap (circular inside the
        # valid range) so positives cannot be solved by state autocorrelation
        # between adjacent transitions. 0 = independent draws.
        self.pair_gap = int(pair_gap)

        # init 0 → s = 1: the policy path starts exactly at the baseline memory.
        # learn_gains=False keeps s ≡ 1 forever (buffer, not Parameter): the
        # policy-path multiply is an exact no-op and the aux reaches only
        # head + encoder — the isolated setting used by the ladder runs.
        if learn_gains:
            self.log_gains = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_buffer("log_gains", torch.zeros(hidden_size))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, proj_dim),
        )

    def gains(self):
        return self.log_gains.exp()

    def forward(
        self,
        z,
        init_hidden,
        init_count,
        cumsum,
        mask=None,
        *,
        detach_gains=False,
        apply_lambda=True,
    ):
        """
        z:           (T, B, D) transition embeddings
        init_hidden: (1, B, D) cumsum seed (init_emb contribution or zeros)
        init_count:  (1, B, 1) initial-memory count
        cumsum:      (T, B, D) seed + running sum of z
        mask:        (T, B, 1) optional validity mask. Padding sits at the tail,
                     so every i ≤ t is valid whenever anchor step t is.
        Returns (scalar loss, info dict of detached GPU tensors). Joint mode
        applies lambda; alternating mode optimizes raw InfoNCE with its own
        learning rate.
        """
        T, B, D = z.shape
        A = self.n_anchors
        if self.detach_inputs:
            z, init_hidden = z.detach(), init_hidden.detach()
            init_count, cumsum = init_count.detach(), cumsum.detach()

        # Bernoulli-masked cumsum for subset/split, via the same cat-cumsum
        # pattern as Mate.forward (avoids pytorch/pytorch#180221). The seed is
        # scaled by beta so the sub-multiset mean stays unbiased for m_t.
        if self.view in ("subset", "split"):
            b = torch.bernoulli(z.new_full((T, B, 1), self.beta))
            csum_b = torch.cat([self.beta * init_hidden, b * z], dim=0).cumsum(dim=0)[1:]

        # A anchor timesteps per episode, t = lo + u^power·(hi-lo) in [frac·L, L-1]
        # (temporal: [frac·L, L-2], keeping ≥1 transition in the suffix).
        # power=1 → uniform; power>1 → early-t oversampling (hard anchors, where
        # the running mean is noisiest and inference speed matters).
        if mask is not None:
            lengths = mask.sum(dim=0).squeeze(-1)  # (B,)
        else:
            lengths = z.new_full((B,), float(T))
        lo = self.min_anchor_frac * lengths
        hi = lengths - (2.0 if self.view == "temporal" else 1.0)
        u = torch.rand((A, B), device=z.device)
        if self.anchor_power != 1.0:
            u = u.pow(self.anchor_power)
        t_idx = (lo + u * (hi - lo).clamp(min=0.0)).long().clamp_(min=0, max=T - 1)  # (A, B)

        idx_d = t_idx.unsqueeze(-1).expand(A, B, D)
        if self.view != "transition":  # transition gathers raw z, no means needed
            cnt = (
                init_count
                + (t_idx + 1).unsqueeze(-1).to(init_count.dtype)
            ).clamp(min=1e-6)                              # (A, B, 1)
            csum_t = torch.gather(cumsum, 0, idx_d)        # (A, B, D)

        if self.view == "transition":
            # Single transitions z_i vs z_j — the loss never touches the memory:
            # if InfoNCE makes single-z context-discriminative, the running mean
            # inherits it by linearity. Initial state and cumsum are unused.
            # Second index set drawn HERE so the RNG order of the other views is
            # untouched (bitwise regression, see brief §2.2).
            u2 = torch.rand((A, B), device=z.device)
            if self.pair_gap > 0:
                # j = i + offset (mod #valid indices), offset ∈ [gap, width-gap]
                # → circular distance ≥ gap ⇒ |i-j| ≥ gap. Approximate when an
                # episode is shorter than 2·gap (the clamp collapses offset to
                # gap). u2 is used RAW here: anchor_power is a prior on anchor
                # POSITION, and applying it to an offset would just bias j
                # toward i+gap.
                width = (hi - lo).clamp(min=0.0) + 1.0                          # (B,)
                off = self.pair_gap + u2 * (width - 2.0 * self.pair_gap).clamp(min=0.0)
                s_idx = (lo + torch.remainder(t_idx.float() - lo + off, width)).long()
                s_idx = s_idx.clamp_(min=0, max=T - 1)
            else:
                # Independent draw, same distribution as t_idx. i == j collisions
                # (prob ~1/L) are just trivially easy positives — left as-is.
                if self.anchor_power != 1.0:
                    u2 = u2.pow(self.anchor_power)
                s_idx = (lo + u2 * (hi - lo).clamp(min=0.0)).long().clamp_(min=0, max=T - 1)
            view_a = torch.gather(z, 0, idx_d)
            view_b = torch.gather(z, 0, s_idx.unsqueeze(-1).expand(A, B, D))
        elif self.view == "prefix":
            # Natural views: two prefix means of the same episode. Nested and
            # leaky (the shorter prefix is a sub-multiset of the longer one);
            # t == s collisions are rare and benign. No masking machinery.
            u2 = torch.rand((A, B), device=z.device)
            if self.anchor_power != 1.0:
                u2 = u2.pow(self.anchor_power)
            s_idx = (lo + u2 * (hi - lo).clamp(min=0.0)).long().clamp_(min=0, max=T - 1)
            csum_s = torch.gather(cumsum, 0, s_idx.unsqueeze(-1).expand(A, B, D))
            cnt_s = (
                init_count
                + (s_idx + 1).unsqueeze(-1).to(init_count.dtype)
            ).clamp(min=1e-6)
            view_a = csum_t / cnt
            view_b = csum_s / cnt_s
        elif self.view == "temporal":
            # "The memory after t steps should already point where the rest of
            # the episode points." The suffix subtraction cancels the init seed
            # exactly, leaving a pure mean over transitions (t, L].
            end_idx = (lengths - 1.0).long().clamp(min=0, max=T - 1)            # (B,) last valid step
            csum_end = torch.gather(cumsum, 0, end_idx.view(1, B, 1).expand(A, B, D))
            cnt_end = init_count + lengths.view(1, B, 1).to(init_count.dtype)
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
        if detach_gains:
            s = s.detach()
        uq = F.normalize(self.head(s * view_a).flatten(0, 1), dim=-1)  # (N, p), N = A·B
        vk = F.normalize(self.head(s * view_b).flatten(0, 1), dim=-1)  # (N, p)
        logits = uq @ vk.t() / self.tau                                # (N, N)
        ep = torch.arange(B, device=logits.device).repeat(A)           # episode id per row
        same_ep = ep.unsqueeze(0) == ep.unsqueeze(1)
        diag = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_ep & ~diag, float("-inf"))
        target = torch.arange(logits.shape[0], device=logits.device)
        ce = F.cross_entropy(logits, target, reduction="none")         # (N,)
        if self.view in ("split", "temporal", "prefix", "transition"):
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
        return (self.lam * loss if apply_lambda else loss), info
