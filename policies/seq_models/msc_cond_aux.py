"""Query-conditioned InfoNCE for MATE's representation step (alternating EMA).

Reference: the "Query-Conditioned MSC" note. A transition splits as
x = (query, response) with query x_b = (obs, act) and response x_t =
(reward, obs-delta / next-obs) — the same [query | response] layout that the
LOO reconstruction uses (`msc_query_dim`, reward always the first response
channel).

The matching task is conditional: given the leave-one-out memory
Z_-j = (S - z_j) / (N - 1) and the anchor's QUERY x_b,j (held fixed), pick
the true RESPONSE x_t,j among the responses of EVERY valid transition in the
batch. This changes what the positive pair shares:

- The shortcut is structurally blocked. All candidates share the anchor's
  query, so context that is visible in the observation (M_obs — e.g.
  Metaworld's rand_vec) contributes nothing to the answer. Episode-level
  InfoNCE could be solved by M_obs alone and regressed on ML10/ML45; here
  the only remaining evidence is how the response depends on the context.
- The loss lower-bounds I(X_t; Z_-j | X_b) — the conditional MI that was
  identified as the policy-relevant quantity S(pi).
- The candidate pool is T*B (not B), lifting the log B ceiling
  (log 64 = 4.16 -> log 12800 ~ 9.5 nats), so saturation is late.

j must be REMOVED from the memory (hence Z_-j): with z_j inside, the
response sits in the encoder's own input and the task is solved by an
identity shortcut, exactly as in the LOO reconstruction.

Known limitation (accepted, see the note): negative responses come from
other queries, so the pair (x_b,j, x_t,j') can sit off the data manifold;
a critic may partially win by detecting physical implausibility rather than
context mismatch. Restricting negatives to query-neighbors is a possible
mitigation, deliberately not implemented in this first version.

Used only through `Mate.contrastive_loss` (alternating_ema mode): a separate
optimizer trains encoder + both towers on this loss, and a frozen EMA
encoder supplies the RL/rollout memory. Never touches the policy path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCCondAux(nn.Module):
    def __init__(
        self,
        hidden_size,
        query_dim,
        response_dim,
        proj_dim=128,
        tau=0.1,
        n_anchors=16,
        hidden=256,
    ):
        super().__init__()
        assert tau > 0.0
        assert int(n_anchors) >= 1
        assert query_dim > 0 and response_dim > 0
        self.tau = float(tau)
        self.n_anchors = int(n_anchors)
        self.query_dim = int(query_dim)

        # Two-tower critic on cosine similarity: s = <f(Z_-j, x_b), g(x_t)> / tau.
        # The note's closed-form Gaussian critic (bilinear + quadratic) showed no
        # gain over this in synthetic CMDPs — the critic form is not the
        # bottleneck; what the pair shares is.
        self.anchor_head = nn.Sequential(
            nn.Linear(hidden_size + query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )
        self.response_head = nn.Sequential(
            nn.Linear(response_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, inputs, z, init_count, cumsum, mask=None):
        """
        inputs:     (T, B, input_size) raw (post-InputNorm) transition tuples
        z:          (T, B, D) online-encoder embeddings (grad flows to encoder)
        init_count: (1, B, 1) initial-memory count (prior weight)
        cumsum:     (T, B, D) prior seed + running sum of z
        mask:       (T, B, 1) optional validity mask (padding at the tail)
        Returns (scalar loss, info dict of detached GPU tensors).
        """
        T, B, D = z.shape
        A = self.n_anchors
        if mask is not None:
            lengths = mask.sum(dim=0).squeeze(-1)  # (B,)
        else:
            lengths = z.new_full((B,), float(T))
        end = (lengths - 1.0).long().clamp(min=0, max=T - 1)

        # Episode totals at the last valid step -> O(1) leave-one-out memory.
        total_z = torch.gather(cumsum, 0, end.view(1, B, 1).expand(1, B, D))
        total_n = init_count + lengths.view(1, B, 1)
        z_loo = (total_z - z) / (total_n - 1.0).clamp(min=1e-6)  # (T, B, D)

        query = inputs[..., : self.query_dim]
        response = inputs[..., self.query_dim:]

        # A anchor steps per episode, uniform over that episode's valid steps.
        u = torch.rand((A, B), device=z.device)
        t_idx = (u * lengths).long().clamp_(min=0, max=T - 1)  # (A, B)
        idx = t_idx.unsqueeze(-1)
        anch_mem = torch.gather(z_loo, 0, idx.expand(A, B, D))
        anch_query = torch.gather(query, 0, idx.expand(A, B, query.shape[-1]))

        u_emb = F.normalize(
            self.anchor_head(torch.cat([anch_mem, anch_query], dim=-1)).flatten(0, 1),
            dim=-1,
        )  # (A*B, p)
        k_emb = F.normalize(
            self.response_head(response).flatten(0, 1), dim=-1
        )  # (T*B, p)
        logits = u_emb @ k_emb.t() / self.tau  # (A*B, T*B)
        if mask is not None:
            invalid = mask.squeeze(-1).flatten() < 0.5  # (T*B,)
            logits = logits.masked_fill(invalid.unsqueeze(0), float("-inf"))
        # Positive = the anchor's own response. flatten(0, 1) maps (t, b) -> t*B + b.
        target = (
            t_idx * B + torch.arange(B, device=z.device).view(1, B)
        ).flatten()  # (A*B,)
        loss = F.cross_entropy(logits, target)

        info = {
            "msc_cond_loss": loss.detach(),
            "msc_cond_acc": (logits.detach().argmax(dim=1) == target).float().mean(),
            "msc_cond_pool": lengths.detach().sum(),  # candidate-pool size (valid transitions)
        }
        return loss, info
