from copy import deepcopy

import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from transformers.activations import ACT2FN
from policies.seq_models.msc_aux import MSCAux
from policies.seq_models.msc_v2_aux import MSCV2Aux

class ResidualFFNBlock(nn.Module):
    """GPT-2 Block minus attention: x + Dropout(W2 · gelu_new(W1 · LN(x))).

    Pre-LN residual feed-forward with inner width 4h and GPT-2 init
    (normal(0, 0.02) weights, zero biases, unit LayerNorm gain).
    """

    def __init__(self, hidden_size, dropout_ff, inner_mult=4,
                 activation="gelu_new", init_std=0.02):
        super().__init__()
        inner = inner_mult * hidden_size
        self.ln = nn.LayerNorm(hidden_size, eps=1e-5)
        self.fc = nn.Linear(hidden_size, inner)
        self.proj = nn.Linear(inner, hidden_size)
        self.act = ACT2FN[activation]
        self.dropout = nn.Dropout(dropout_ff)
        for lin in (self.fc, self.proj):
            nn.init.normal_(lin.weight, mean=0.0, std=init_std)
            nn.init.zeros_(lin.bias)

    def forward(self, x):
        return x + self.dropout(self.proj(self.act(self.fc(self.ln(x)))))


def build_mate_embedder(embedder_type, input_size, hidden_size, n_layer,
                        dropout_emb, dropout_ff):
    """transition_size -> hidden_size pipeline used by Mate.

    Both start with the same input projection as RNN_head's transition_embedder
    for non-MATE models: Linear(in->h) -> LeakyReLU -> Dropout(dropout_emb).
      mlp     : n_layer x (Linear(h->h) -> LeakyReLU -> Dropout(dropout_ff))
      gpt_ffn : Dropout(dropout_emb) (GPT2Model.drop), then
                n_layer x ResidualFFNBlock (pre-LN, 4h, gelu_new, resid dropout)
                followed by a final LayerNorm (GPT-2's ln_f), i.e. GPT-2 with the
                positional encoding and attention removed; mean aggregation then
                takes attention's place.
    """
    layers = [
        nn.Linear(input_size, hidden_size),
        nn.LeakyReLU(),
        nn.Dropout(dropout_emb),
    ]
    if embedder_type == "mlp":
        for _ in range(n_layer):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_ff),
            ]
    elif embedder_type == "gpt_ffn":
        layers += [ResidualFFNBlock(hidden_size, dropout_ff) for _ in range(n_layer)]
        layers.append(nn.LayerNorm(hidden_size, eps=1e-5))
    else:
        raise ValueError(
            f"embedder_type must be 'mlp' or 'gpt_ffn', got {embedder_type!r}"
        )
    return nn.Sequential(*layers)


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, dropout_ff=0.05, dropout_emb=0.05, learn_init_emb=False, use_ema_init_emb=False, ema_init_emb_beta=5e-4, use_store=False, store_grad_correction=True, store_fresh_target=True, store_independent_loss_rows=False, msc_enable=False, msc_objective="legacy", msc_lambda=0.1, msc_beta=0.7, msc_tau=0.1, msc_k_min=8, msc_k_max=64, msc_n_anchors=4, msc_proj_dim=128, msc_min_anchor_frac=0.1, msc_detach_z=True, msc_view="subset", msc_focal_gamma=0.0, msc_anchor_power=1.0, msc_learn_gains=True, msc_pair_gap=0, msc_update_mode="joint", embedder_type="mlp", **kwargs):
        super().__init__()
        # input_size = raw transition_size (post-InputNorm); RNN_head sets transition_embedder=Identity for mate.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.msc_objective = msc_objective
        if self.msc_objective not in ("legacy", "v2"):
            raise ValueError("msc_objective must be 'legacy' or 'v2'")
        self.msc_update_mode = msc_update_mode
        if self.msc_update_mode not in ("joint", "alternating_ema"):
            raise ValueError(
                "msc_update_mode must be 'joint' or 'alternating_ema'"
            )
        self.alternating_msc = self.msc_update_mode == "alternating_ema"

        # One input projection followed by n_layer additional blocks
        # (plain MLP layers or GPT-2 residual FFN blocks; see build_mate_embedder).
        self.embedder_type = embedder_type
        self.embedder = build_mate_embedder(
            embedder_type, input_size, hidden_size, n_layer, dropout_emb, dropout_ff,
        )

        print(
            f"Mate embedder: type={embedder_type}, n_layer={n_layer}, "
            f"input_size={input_size}, hidden_size={hidden_size}"
        )

        # Initial-memory prior: m_t = (w * init_emb + sum_i E(x_i)) / (w + t),
        # where init_emb is learned or tracked as an EMA and w is always learned.
        self.learn_init_emb = learn_init_emb
        self.use_ema_init_emb = use_ema_init_emb
        self.ema_init_emb_beta = float(ema_init_emb_beta)
        # STORE (Subset Training Over REused Embeddings): recompute only the
        # sampled transitions and reuse cached embeddings for the rest.
        self.use_store = bool(use_store)
        self.store_grad_correction = bool(store_grad_correction)
        self.store_independent_loss_rows = bool(store_independent_loss_rows)
        # False: the successor memory (target input) uses only cached z.
        self.store_fresh_target = bool(store_fresh_target)
        if self.use_store and msc_enable:
            raise ValueError("use_store (STORE) is not supported with MSC")
        if self.use_ema_init_emb and not self.learn_init_emb:
            raise ValueError("use_ema_init_emb requires learn_init_emb=True")
        if self.use_ema_init_emb and not 0.0 < self.ema_init_emb_beta <= 1.0:
            raise ValueError("ema_init_emb_beta must be in (0, 1]")
        if self.learn_init_emb:
            if self.use_ema_init_emb:
                self.register_buffer("init_emb", torch.zeros(self.hidden_size))
                self.register_buffer("_ema_init_emb_t", torch.zeros(()))
            else:
                self.init_emb = nn.Parameter(ptu.randn(self.hidden_size))
            self.log_init_weight = nn.Parameter(ptu.zeros(()))

        # MSC contrastive aux (see msc_aux.py). Joint mode adds its loss to the
        # RL backward; alternating_ema trains the online embedder separately
        # and uses a frozen EMA copy on the policy path.
        if not msc_enable:
            self.msc = None
        elif self.msc_objective == "v2":
            self.msc = MSCV2Aux(
                hidden_size,
                msc_lambda=msc_lambda,
                tau=msc_tau,
                k_min=msc_k_min,
                k_max=msc_k_max,
                detach_z=msc_detach_z,
            )
        else:
            self.msc = MSCAux(
                hidden_size, msc_lambda=msc_lambda, beta=msc_beta, tau=msc_tau,
                n_anchors=msc_n_anchors, proj_dim=msc_proj_dim,
                min_anchor_frac=msc_min_anchor_frac, detach_inputs=msc_detach_z,
                view=msc_view, focal_gamma=msc_focal_gamma, anchor_power=msc_anchor_power,
                learn_gains=msc_learn_gains, pair_gap=msc_pair_gap,
            )
        if self.alternating_msc:
            if self.msc is None:
                raise ValueError("alternating_ema requires msc_enable=True")
            if msc_detach_z:
                raise ValueError(
                    "alternating_ema requires msc_detach_z=False"
                )
            if not any(p.requires_grad for p in self.embedder.parameters()):
                raise ValueError(
                    "alternating_ema requires a trainable Mate.embedder"
                )
            self.ema_embedder = deepcopy(self.embedder)
            self.ema_embedder.requires_grad_(False)
            self.ema_embedder.eval()
        else:
            self.ema_embedder = None
        if self.msc is not None:
            print(
                "Using MSC in Mate: "
                f"objective={self.msc_objective}, lambda={msc_lambda}, "
                f"tau={msc_tau}, detach_z={msc_detach_z}"
            )

    def train(self, mode=True):
        super().train(mode)
        if self.ema_embedder is not None:
            self.ema_embedder.eval()
        return self

    def embed_transitions(self, inputs):
        embedder = (
            self.ema_embedder if self.alternating_msc else self.embedder
        )
        return embedder(inputs)

    def forward(
        self, inputs, h_0, mask=None, compute_msc=True,
        **kwargs,
    ):
        """
        inputs: (T, B, input_size)
        h_0: (1, B, hidden_size), (1, B, 1)   # cumulative sum, count
        mask: optional (T, B, 1) validity mask (training only; consumed by MSC anchor sampling)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), (1, B, 1)
        """
        hidden, initial_count = h_0
        z = self.embed_transitions(inputs)  # (L, B, hidden_size)
        info = {}

        # cat([init, x]).cumsum(dim=0)[1:] == init + x.cumsum(dim=0)
        # avoids Inductor SplitScan + broadcast crash (pytorch/pytorch#180221)
        cumsum = torch.cat([hidden, z], dim=0).cumsum(dim=0)[1:]
        step_counts = torch.arange(
            1,
            z.shape[0] + 1,
            device=initial_count.device,
            dtype=initial_count.dtype,
        ).view(-1, 1, 1)
        counts = initial_count + step_counts
        h_n = cumsum[-1].clone().unsqueeze(0)
        count_n = counts[-1].clone().unsqueeze(0)
        output = cumsum / counts.clamp(min=1e-6) # (L, B, hidden_size)

        # Joint MSC computes InfoNCE here. Alternating MSC computes it through
        # contrastive_loss() and this path consumes only the EMA embedder.
        # Learned gains are applied to the running memory before positional
        # encoding.
        if self.msc is not None:
            if compute_msc and self.training and not self.alternating_msc:
                if self.msc_objective == "v2":
                    msc_loss, msc_info = self.msc(
                        z=z,
                        init_hidden=hidden,
                        init_count=initial_count,
                        mask=mask,
                    )
                else:
                    msc_loss, msc_info = self.msc(
                        z=z, init_hidden=hidden,
                        init_count=initial_count, cumsum=cumsum, mask=mask,
                    )
                info["_aux_loss"] = msc_loss
                info.update(msc_info)
            if self.msc_objective == "legacy":
                output = self.msc.gains() * output

        info.update(self._embedding_info(z, mask))

        return output, (h_n, count_n), info

    def forward_cached(
        self,
        inputs,
        h_0,
        cached_embeddings,
        embed_t,
        loss_cached,
        loss_prefixes,
        transition_t,
        mask=None,
    ):
        """
        inputs / cached_embeddings / embed_t: the sorted rows whose embeddings
        are recomputed. loss_cached / loss_prefixes / transition_t: the rows the
        losses are computed at (the same rows unless store_independent_loss_rows).
        """
        hidden, initial_count = h_0
        z = self.embed_transitions(inputs)
        loss_cached = loss_cached.to(z)
        loss_prefixes = loss_prefixes.to(z)

        # delta_sums[j] = sum of the first j recomputed deltas; searchsorted counts
        # the recomputed rows before (right=False) / up to (right=True) each loss row.
        delta = z - cached_embeddings.to(z)
        delta_sums = torch.cat((torch.zeros_like(delta[:1]), delta.cumsum(dim=0)), dim=0)
        embed_rows = embed_t.T.contiguous()
        loss_rows = transition_t.T.contiguous()

        def _delta_sum(right):
            n = torch.searchsorted(embed_rows, loss_rows, right=right).T
            return delta_sums.gather(0, n.unsqueeze(-1).expand(-1, -1, z.shape[-1]))

        correction_before = _delta_sum(right=False)
        # correction_before is the ONLY gradient path from the loss to z (next_joint
        # is consumed under no_grad in both agents), so every surviving pair has
        # i < t. Paired rows need BOTH in one k-subset: p = k(k-1)/(T(T-1));
        # independent rows: p = (k/T)^2. Paths that skip z need one row: p = k/T.
        # Under the shared 1/num_valid loss normalization the embedder is therefore
        # scaled down by (k-1)/(T-1) (paired) or k/T (independent); undo it with a
        # straight-through factor that leaves the forward value untouched.
        # Sound in window mode too: that path is reachable only without truncation,
        # where k == T and alpha == 1.
        if self.store_grad_correction and z.shape[0] > 1:
            T, k = self.max_seq_length - 1, z.shape[0]
            alpha = T / k if self.store_independent_loss_rows else (T - 1) / (k - 1)
            frozen = correction_before.detach()
            correction_before = frozen + alpha * (correction_before - frozen)
        current_sums = hidden + loss_prefixes + correction_before
        next_sums = hidden + loss_prefixes + loss_cached
        if self.store_fresh_target:
            next_sums = next_sums + _delta_sum(right=True)

        physical_steps = transition_t.to(initial_count).unsqueeze(-1)
        current_counts = initial_count + physical_steps - 1.0
        next_counts = initial_count + physical_steps
        current_output = current_sums / current_counts.clamp(min=1e-6)
        next_output = next_sums / next_counts.clamp(min=1e-6)

        return (
            current_output,
            next_output,
            self._embedding_info(z, mask),
            z.detach(),
        )

    def _embedding_info(self, z, mask):
        info = {}
        self._update_ema_init_emb(z, mask)
        if self.learn_init_emb:
            info["init_emb_norm"] = self.init_emb.detach().norm()
            info["init_weight"] = self.log_init_weight.detach().exp()
        return info

    @torch.no_grad()
    def _update_ema_init_emb(self, z, mask):
        if not self.use_ema_init_emb or not self.training:
            return
        if mask is None:
            mask = z.new_ones((*z.shape[:2], 1))
        else:
            mask = mask.to(z.dtype)
        total = mask.sum()
        has_valid = total > 0
        mean = (z.detach() * mask).sum((0, 1)) / total.clamp_min(1.0)
        next_t = self._ema_init_emb_t + has_valid.to(self._ema_init_emb_t.dtype)
        beta_t = self.ema_init_emb_beta / (
            1.0 - (1.0 - self.ema_init_emb_beta) ** next_t.clamp_min(1.0)
        )
        next_init_emb = (
            (1.0 - beta_t) * self.init_emb + beta_t * mean
        )
        self.init_emb.copy_(torch.where(has_valid, next_init_emb, self.init_emb))
        self._ema_init_emb_t.copy_(next_t)

    def contrastive_loss(self, inputs, h_0, mask=None):
        if not self.alternating_msc:
            raise RuntimeError(
                "contrastive_loss is only available in alternating_ema mode"
            )

        hidden, initial_count = (state.detach() for state in h_0)
        inputs = inputs.detach()
        z = self.embedder(inputs)

        if self.msc_objective == "v2":
            return self.msc(
                z=z,
                init_hidden=hidden,
                init_count=initial_count,
                mask=mask,
                apply_lambda=False,
            )

        cumsum = torch.cat([hidden, z], dim=0).cumsum(dim=0)[1:]
        return self.msc(
            z=z,
            init_hidden=hidden,
            init_count=initial_count,
            cumsum=cumsum,
            mask=mask,
            detach_gains=True,
            apply_lambda=False,
        )

    def msc_parameters(self):
        if not self.alternating_msc:
            return ()
        aux_parameters = (
            self.msc.parameters()
            if self.msc_objective == "v2"
            else self.msc.head.parameters()
        )
        return tuple(self.embedder.parameters()) + tuple(aux_parameters)

    @torch.no_grad()
    def update_msc_ema(self, tau):
        if not self.alternating_msc:
            return
        if not 0.0 < tau <= 1.0:
            raise ValueError("EMA tau must be in (0, 1]")
        ptu.soft_update_from_to(self.embedder, self.ema_embedder, tau)

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        """Internal state: (cumulative sum, count)."""
        if self.learn_init_emb:
            t_0 = self.log_init_weight.exp().view(1, 1, 1).expand(1, batch_size, 1)
            init_emb = self.init_emb.clone() if self.use_ema_init_emb else self.init_emb
            h_0 = init_emb.view(1, 1, -1).expand(1, batch_size, -1) * t_0
        else:
            t_0 = ptu.zeros((1, batch_size, 1))
            h_0 = ptu.zeros((1, batch_size, self.hidden_size))
        return h_0, t_0

    def internal_state_to_hidden(self, internal_state):
        # Mirrors the forward output: running mean (⊙ MSC gains).
        hidden, count = internal_state
        out = hidden / count.clamp(min=1e-6)
        if self.msc is not None and self.msc_objective == "legacy":
            out = self.msc.gains() * out
        return out
