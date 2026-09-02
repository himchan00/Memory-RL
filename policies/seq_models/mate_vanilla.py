from copy import deepcopy

import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp
from policies.seq_models.Rff_embedding import RFFEmbedding
from policies.seq_models.msc_aux import MSCAux
from policies.seq_models.msc_v2_aux import MSCV2Aux
from policies.seq_models.msc_v3_aux import MSCV3Aux
from policies.seq_models.msc_cond_aux import MSCCondAux
from policies.seq_models.msc_unicorn_aux import MSCUnicornAux


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, dropout_ff=0.05, dropout_emb=0.05, use_rff=False, kernel="gaussian", learn_kernel="off", learn_init_emb=False, use_ema_init_emb=False, ema_init_emb_beta=5e-4, msc_enable=False, msc_objective="legacy", msc_lambda=0.1, msc_beta=0.7, msc_tau=0.1, msc_k_min=8, msc_k_max=64, msc_n_anchors=4, msc_proj_dim=128, msc_min_anchor_frac=0.1, msc_detach_z=True, msc_view="subset", msc_focal_gamma=0.0, msc_anchor_power=1.0, msc_learn_gains=True, msc_pair_gap=0, msc_update_mode="joint", msc_nce_weight=1.0, msc_recon_beta=0.0, msc_recon_hidden=256, msc_recon_target="all", msc_recon_decoder="mlp", msc_recon_ridge=1e-3, msc_query_dim=None, **kwargs):
        super().__init__()
        # input_size = raw transition_size (post-InputNorm); RNN_head sets transition_embedder=Identity for mate.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.msc_objective = msc_objective
        if self.msc_objective not in ("legacy", "v2", "v3", "conditional", "unicorn"):
            raise ValueError(
                "msc_objective must be 'legacy', 'v2', 'v3', 'conditional' or 'unicorn'"
            )
        self.msc_update_mode = msc_update_mode
        if self.msc_update_mode not in ("joint", "alternating_ema", "alternating_online"):
            raise ValueError(
                "msc_update_mode must be 'joint', 'alternating_ema' or "
                "'alternating_online'"
            )
        # Both alternating modes run the 2-step machinery (separate msc
        # optimizer, msc_updates_per_rl, own lr/clipping). They differ in who
        # shapes the encoder the policy consumes:
        #   alternating_ema    -- RL sees a frozen EMA encoder; the rep loss is
        #                         the ONLY force on the representation.
        #   alternating_online -- RL sees (and trains) the online encoder; the
        #                         rep loss is a decoupled auxiliary force.
        self.alternating_msc = self.msc_update_mode in (
            "alternating_ema", "alternating_online"
        )

        # Embedder: (n_layer + 1) blocks total = 1 input projection (in→h) + n_layer additional (h→h).
        layers = []
        for i in range(n_layer + 1):
            is_first = (i == 0)
            is_last = (i == n_layer)
            in_dim = input_size if is_first else hidden_size
            if is_last and use_rff:
                # if use_rff, last layer becomes RFF embedding
                layers.append(RFFEmbedding(input_dim=in_dim, embedding_dim=hidden_size, kernel=kernel, learn_kernel=learn_kernel))
            else:
                # First block uses dropout_emb (input projection); rest use dropout_ff.
                layers += [nn.Linear(in_dim, hidden_size), nn.LeakyReLU(),
                           nn.Dropout(dropout_emb if is_first else dropout_ff)]
        self.embedder = nn.Sequential(*layers)
        self._rff_layer = self.embedder[-1] if (use_rff and isinstance(self.embedder[-1], RFFEmbedding)) else None

        print(f"Mate embedder: use_rff={use_rff}, n_layer={n_layer}, input_size={input_size}, hidden_size={hidden_size}, learn_kernel={learn_kernel}")

        # Initial-memory prior: m_t = (w * init_emb + sum_i E(x_i)) / (w + t),
        # where init_emb is learned or tracked as an EMA and w is always learned.
        self.learn_init_emb = learn_init_emb
        self.use_ema_init_emb = use_ema_init_emb
        self.ema_init_emb_beta = float(ema_init_emb_beta)
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
        elif self.msc_objective == "v3":
            # Scale-covering episode CPC (see msc_v3_aux.py). Alternating-only:
            # the joint-mode forward hook has no v3 signature, and the note's
            # analysis assumes the representation step owns the encoder.
            if not self.alternating_msc:
                raise ValueError(
                    "msc_objective='v3' requires an alternating msc_update_mode"
                )
            if msc_query_dim is None:
                raise ValueError(
                    "msc_objective='v3' requires msc_query_dim "
                    "(RNN_head passes it; needed by the query-twin diagnostic)"
                )
            self.msc = MSCV3Aux(
                hidden_size,
                query_dim=msc_query_dim,
                msc_lambda=msc_lambda,
                tau=msc_tau,
                twin_correction=bool(kwargs.get("msc_v3_twin_correction", False)),
                detach_z=msc_detach_z,
            )
            print(
                f"Using scale-covering CPC (v3) in Mate: tau={msc_tau}, "
                f"query_dim={msc_query_dim}, "
                f"twin_correction={bool(kwargs.get('msc_v3_twin_correction', False))}"
            )
        elif self.msc_objective == "conditional":
            # Query-conditioned InfoNCE (see msc_cond_aux.py). Only meaningful
            # as a representation-step objective: it exists to shape the
            # encoder, and Mate.forward has no joint-mode hook for it.
            if not self.alternating_msc:
                raise ValueError(
                    "msc_objective='conditional' requires an alternating msc_update_mode"
                )
            if msc_query_dim is None:
                raise ValueError(
                    "msc_objective='conditional' requires msc_query_dim "
                    "(RNN_head passes it; the transition tuple is [query | response])"
                )
            if not 0 < msc_query_dim < input_size:
                raise ValueError(
                    f"msc_query_dim must be in (0, {input_size}), got {msc_query_dim}"
                )
            self.msc = MSCCondAux(
                hidden_size,
                query_dim=msc_query_dim,
                response_dim=input_size - msc_query_dim,
                proj_dim=msc_proj_dim,
                tau=msc_tau,
                n_anchors=msc_n_anchors,
                hidden=int(msc_recon_hidden),
            )
            print(
                f"Using conditional InfoNCE in Mate: tau={msc_tau}, "
                f"n_anchors={msc_n_anchors}, proj_dim={msc_proj_dim}, "
                f"query_dim={msc_query_dim}, response_dim={input_size - msc_query_dim}"
            )
        elif self.msc_objective == "unicorn":
            # UNICORN-SS (see msc_unicorn_aux.py): L_recon + a/(1-a) * L_FOCAL.
            # Works in ALL update modes — joint feeds lambda*L into the RL
            # backward; alternating modes optimize the raw L separately.
            if msc_query_dim is None:
                raise ValueError(
                    "msc_objective='unicorn' requires msc_query_dim "
                    "(RNN_head passes it; the transition tuple is [query | response])"
                )
            if not 0 < msc_query_dim < input_size:
                raise ValueError(
                    f"msc_query_dim must be in (0, {input_size}), got {msc_query_dim}"
                )
            unicorn_alpha = float(kwargs.get("msc_unicorn_alpha", 0.5))
            self.msc = MSCUnicornAux(
                hidden_size,
                query_dim=msc_query_dim,
                response_dim=input_size - msc_query_dim,
                msc_lambda=msc_lambda,
                alpha=unicorn_alpha,
                recon_hidden=int(msc_recon_hidden),
                detach_z=msc_detach_z,
            )
            print(
                f"Using UNICORN in Mate: alpha={unicorn_alpha} "
                f"(focal_coef={unicorn_alpha / (1 - unicorn_alpha):.3f}), "
                f"lambda={msc_lambda}, query_dim={msc_query_dim}, "
                f"response_dim={input_size - msc_query_dim}, detach_z={msc_detach_z}"
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
                raise ValueError("alternating MSC requires msc_enable=True")
            if msc_detach_z:
                raise ValueError(
                    "alternating MSC requires msc_detach_z=False"
                )
            if not any(p.requires_grad for p in self.embedder.parameters()):
                raise ValueError(
                    "alternating MSC requires a trainable Mate.embedder"
                )
        if self.msc_update_mode == "alternating_ema":
            self.ema_embedder = deepcopy(self.embedder)
            self.ema_embedder.requires_grad_(False)
            self.ema_embedder.eval()
        else:
            # alternating_online: RL consumes and trains the online embedder
            # directly; there is no EMA copy (update_msc_ema is a no-op).
            self.ema_embedder = None

        # --- representation-step loss weights (alternating_ema only) ---------
        # L_rep = msc_nce_weight * InfoNCE + msc_recon_beta * LOO-reconstruction.
        # (1, 0) reproduces the pure-contrastive behaviour bit-for-bit;
        # (0, 1) is the pure predictive-sufficiency objective.
        self.msc_nce_weight = float(msc_nce_weight)
        self.msc_recon_beta = float(msc_recon_beta)
        self.msc_recon_target = msc_recon_target
        self.msc_query_dim = msc_query_dim
        self.msc_decoder = None
        self.msc_recon_decoder = msc_recon_decoder
        self.msc_recon_ridge = float(msc_recon_ridge)
        self._recon_enabled = False
        self._recon_cond_checked = False
        if self.alternating_msc:
            if self.msc_nce_weight < 0.0 or self.msc_recon_beta < 0.0:
                raise ValueError("msc_nce_weight / msc_recon_beta must be >= 0")
            if self.msc_nce_weight <= 0.0 and self.msc_recon_beta <= 0.0:
                raise ValueError(
                    "alternating_ema needs a nonzero msc_nce_weight or msc_recon_beta"
                )
        if self.alternating_msc and self.msc_recon_beta > 0.0:
            self._recon_enabled = True
            if msc_query_dim is None:
                raise ValueError(
                    "msc_recon_beta > 0 requires msc_query_dim "
                    "(RNN_head passes it; the transition tuple is [query | response])"
                )
            if not 0 < msc_query_dim < input_size:
                raise ValueError(
                    f"msc_query_dim must be in (0, {input_size}), got {msc_query_dim}"
                )
            # transition tuple layout is [query | reward | rest-of-response],
            # so the reward is always the first response channel.
            if msc_recon_target == "all":
                self._recon_slice = slice(msc_query_dim, input_size)
            elif msc_recon_target == "reward":
                self._recon_slice = slice(msc_query_dim, msc_query_dim + 1)
            elif msc_recon_target == "dynamics":
                self._recon_slice = slice(msc_query_dim + 1, input_size)
            else:
                raise ValueError(
                    "msc_recon_target must be 'all', 'reward' or 'dynamics'"
                )
            target_dim = self._recon_slice.stop - self._recon_slice.start
            if target_dim <= 0:
                raise ValueError(
                    f"msc_recon_target={msc_recon_target!r} selects an empty slice "
                    "for this transition layout"
                )
            if msc_recon_decoder not in ("mlp", "linear"):
                raise ValueError("msc_recon_decoder must be 'mlp' or 'linear'")
            if msc_recon_decoder == "mlp":
                # Learned throwaway decoder, same scale as the InfoNCE projection
                # head (~1.2x its parameter count). Never touches the policy path.
                self.msc_decoder = Mlp(
                    input_size=hidden_size + msc_query_dim,
                    output_size=target_dim,
                    hidden_sizes=[int(msc_recon_hidden)],
                    output_activation='linear',
                    dropout=dropout_ff,
                )
                extra = f"hidden={int(msc_recon_hidden)}"
            else:
                # No decoder parameters at all: the readout is the ridge-regression
                # solution for this batch, recomputed every forward. Requiring the
                # response to be LINEARLY readable from (memory, query) is exactly
                # MATE's design premise -- the posterior is supposed to depend on
                # the data only through the mean embedding.
                self.msc_decoder = None
                extra = f"ridge={self.msc_recon_ridge:g}"
            print(
                f"Using MSC reconstruction in Mate: beta={self.msc_recon_beta}, "
                f"nce_weight={self.msc_nce_weight} ({self.msc_objective}), "
                f"decoder={msc_recon_decoder}, "
                f"target={msc_recon_target}, query_dim={msc_query_dim}, "
                f"target_dim={target_dim}, {extra}"
            )
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

    def forward(self, inputs, h_0, mask=None, compute_msc=True, **kwargs):
        """
        inputs: (T, B, input_size)
        h_0: (1, B, hidden_size), (1, B, 1)   # cumulative sum, count
        mask: optional (T, B, 1) validity mask (training only; consumed by MSC anchor sampling)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), (1, B, 1)
        """
        hidden, initial_count = h_0
        embedder = (
            self.ema_embedder
            if self.msc_update_mode == "alternating_ema"
            else self.embedder
        )
        z = embedder(inputs) # (L, B, hidden_size)
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
        # Spectral gains s are applied pre-PE;
        # by linearity s ⊙ m_t equals the running mean of s ⊙ z — a diagonal
        # reweighting of the kernel's spectral measure (Prop 5), so the memory
        # stays a kernel mean embedding under the reweighted kernel.
        if self.msc is not None:
            if compute_msc and self.training and not self.alternating_msc:
                if self.msc_objective == "v2":
                    msc_loss, msc_info = self.msc(
                        z=z,
                        init_hidden=hidden,
                        init_count=initial_count,
                        mask=mask,
                    )
                elif self.msc_objective == "unicorn":
                    # Joint-mode UNICORN: lambda-weighted into the RL backward.
                    msc_loss, msc_info = self.msc(
                        inputs=inputs, z=z, init_hidden=hidden,
                        init_count=initial_count, cumsum=cumsum, mask=mask,
                        apply_lambda=True,
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

        if self._rff_layer is not None:
            info.update(self._rff_layer.logging_stats())

        self._update_ema_init_emb(z, mask)
        if self.learn_init_emb:
            info["init_emb_norm"] = self.init_emb.detach().norm()
            info["init_weight"] = self.log_init_weight.detach().exp()

        return output, (h_n, count_n), info

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

        # All objectives share one weighting scheme:
        #   L_rep = msc_nce_weight * InfoNCE + msc_recon_beta * LOO-reconstruction
        # v2 builds its own subset means internally, so it never needs the
        # running sums; legacy, conditional and the reconstruction term do.
        need_running_sums = self._recon_enabled or (
            self.msc_nce_weight > 0.0
            and self.msc_objective in ("legacy", "conditional", "unicorn")
        )
        if need_running_sums:
            cumsum = torch.cat([hidden, z], dim=0).cumsum(dim=0)[1:]

        loss = None
        info = {}
        if self.msc_nce_weight > 0.0:
            if self.msc_objective == "v2":
                nce_loss, info = self.msc(
                    z=z,
                    init_hidden=hidden,
                    init_count=initial_count,
                    mask=mask,
                    apply_lambda=False,
                )
            elif self.msc_objective == "v3":
                nce_loss, info = self.msc(
                    inputs=inputs,
                    z=z,
                    init_hidden=hidden,
                    init_count=initial_count,
                    mask=mask,
                    apply_lambda=False,
                )
            elif self.msc_objective == "conditional":
                nce_loss, info = self.msc(
                    inputs=inputs,
                    z=z,
                    init_count=initial_count,
                    cumsum=cumsum,
                    mask=mask,
                )
            elif self.msc_objective == "unicorn":
                nce_loss, info = self.msc(
                    inputs=inputs,
                    z=z,
                    init_hidden=hidden,
                    init_count=initial_count,
                    cumsum=cumsum,
                    mask=mask,
                    apply_lambda=False,
                )
            else:
                nce_loss, info = self.msc(
                    z=z,
                    init_hidden=hidden,
                    init_count=initial_count,
                    cumsum=cumsum,
                    mask=mask,
                    detach_gains=True,
                    apply_lambda=False,
                )
            loss = self.msc_nce_weight * nce_loss

        if self._recon_enabled:
            recon = self._recon_loss(inputs, z, initial_count, cumsum, mask)
            loss = recon * self.msc_recon_beta if loss is None else loss + self.msc_recon_beta * recon
            info["msc_recon"] = recon.detach()

        return loss, info

    def _recon_loss(self, inputs, z, initial_count, cumsum, mask):
        """Leave-one-out predictive reconstruction (predictive-sufficiency objective).

        For every valid step j the memory is rebuilt WITHOUT j -- O(1) per step,
        because the memory is a running mean over unit-weight embeddings:

            Z_-j = (S - z_j) / (N - 1),   S = cumsum[end], N = init_count + L

        and a decoder predicts the response x_t,j from (Z_-j, x_b,j). Predicting
        x_t,j from z_j itself would be solved by the identity map (the response is
        inside the encoder's own input), which is why j must be removed.

        Every valid transition is used as a target -- there is no anchor count to
        tune, and the cost stays O(T) because no pairwise term is involved.
        """
        T_len, B, D = z.shape
        if mask is not None:
            lengths = mask.sum(dim=0).squeeze(-1)                       # (B,)
            end = (lengths - 1.0).long().clamp(min=0, max=T_len - 1)    # (B,)
        else:
            lengths = z.new_full((B,), float(T_len))
            end = torch.full((B,), T_len - 1, device=z.device, dtype=torch.long)

        # Episode totals, gathered at the last valid step (padding sits at the tail).
        total_z = torch.gather(cumsum, 0, end.view(1, B, 1).expand(1, B, D))   # (1,B,D)
        total_n = initial_count + lengths.view(1, B, 1)                        # (1,B,1)

        z_loo = (total_z - z) / (total_n - 1.0).clamp(min=1e-6)                # (T,B,D)
        query = inputs[..., :self.msc_query_dim]
        target = inputs[..., self._recon_slice]

        if self.msc_recon_decoder == "linear":
            return self._recon_linear(z_loo, query, target, mask)

        pred = self.msc_decoder(torch.cat([z_loo, query], dim=-1))
        se = (pred - target).pow(2).mean(dim=-1, keepdim=True)                 # (T,B,1)
        if mask is None:
            return se.mean()
        return (se * mask).sum() / mask.sum().clamp(min=1.0)

    def _recon_linear(self, z_loo, query, target, mask):
        """Decoder-free readout: ridge regression solved in closed form per batch.

        The readout weights are a FUNCTION OF THE BATCH, not parameters an
        optimizer owns, so this variant adds zero learnable modules. Demanding a
        LINEAR readout of the response from (memory, query) is exactly MATE's
        design premise -- the posterior is meant to depend on the data only
        through the mean embedding.

        Episodes are split in half: the weights are fit on one half and scored on
        the other (both directions, averaged). Scoring the same rows used for the
        fit would reward the encoder for supplying directions that fit noise; the
        split removes that incentive and makes the reported value an honest
        held-out residual.
        """
        T, B, _ = z_loo.shape
        ones = torch.ones_like(query[..., :1])
        feat = torch.cat([z_loo, query, ones], dim=-1)                # (T,B,d)
        w = torch.ones_like(ones) if mask is None else mask           # (T,B,1)

        def solve(f, y, wt):
            fm = (f * wt).reshape(-1, f.shape[-1])                    # (n,d)
            ym = (y * wt).reshape(-1, y.shape[-1])                    # (n,m)
            gram = fm.transpose(0, 1) @ fm
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            gram = gram + self.msc_recon_ridge * gram.diagonal().mean().clamp(min=1e-8) * eye
            return torch.linalg.solve(gram, fm.transpose(0, 1) @ ym)  # (d,m)

        def score(f, y, wt, weights):
            se = (f @ weights - y).pow(2).mean(dim=-1, keepdim=True)
            return (se * wt).sum() / wt.sum().clamp(min=1.0)

        half = B // 2
        # The fit is only meaningful when rows comfortably outnumber columns.
        # mujoco (T=200, B=64) gives n/d ~ 17-23; short episodes can fall near 1,
        # where the solution interpolates and the residual stops being informative.
        if not self._recon_cond_checked:
            self._recon_cond_checked = True
            ratio = (T * max(half, 1)) / feat.shape[-1]
            if ratio < 5.0:
                print(
                    f"[MSC] warning: linear readout is poorly determined "
                    f"(rows/cols = {ratio:.1f}); raise msc_recon_ridge, enlarge the "
                    f"batch, or use msc_recon_decoder=mlp"
                )
        if half == 0:                       # single episode: no split possible
            return score(feat, target, w, solve(feat, target, w))

        parts = ((slice(0, half), slice(half, B)), (slice(half, B), slice(0, half)))
        total = None
        for fit, ev in parts:
            weights = solve(feat[:, fit], target[:, fit], w[:, fit])
            s = score(feat[:, ev], target[:, ev], w[:, ev], weights)
            total = s if total is None else total + s
        return total / len(parts)

    def msc_parameters(self):
        if not self.alternating_msc:
            return ()
        params = tuple(self.embedder.parameters())
        # The InfoNCE-side parameters (v2: bilinear weight + log_kappa;
        # conditional: the two towers; legacy: the projection head) get no
        # gradient when msc_nce_weight == 0, so leave them out of the optimizer
        # entirely -- otherwise a pure reconstruction run silently carries dead
        # weights. These never sit on the policy path: legacy gains (log_gains)
        # are a separate parameter owned by the RL optimizer, and v2/conditional
        # have no gains.
        if self.msc_nce_weight > 0.0:
            params += tuple(
                self.msc.head.parameters()
                if self.msc_objective == "legacy"
                else self.msc.parameters()
            )
        if self.msc_decoder is not None:
            params += tuple(self.msc_decoder.parameters())
        return params

    def msc_exclusive_parameters(self):
        """Parameters owned ONLY by the msc optimizer (excluded from RL).

        alternating_ema: everything msc trains, embedder included — RL must
        not touch the encoder (it consumes the frozen EMA copy instead).
        alternating_online: only the aux-specific heads — the embedder is
        deliberately shared: RL trains it through the policy path and the rep
        step trains it through the msc loss, each with its own optimizer.
        """
        if not self.alternating_msc:
            return ()
        if self.msc_update_mode == "alternating_ema":
            return self.msc_parameters()
        embedder_ids = {id(p) for p in self.embedder.parameters()}
        return tuple(
            p for p in self.msc_parameters() if id(p) not in embedder_ids
        )

    @torch.no_grad()
    def update_msc_ema(self, tau):
        if not self.alternating_msc or self.ema_embedder is None:
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
