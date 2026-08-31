from copy import deepcopy

import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from policies.seq_models.Rff_embedding import RFFEmbedding
from policies.seq_models.msc_aux import MSCAux
from policies.seq_models.msc_v2_aux import MSCV2Aux


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, dropout_ff=0.05, dropout_emb=0.05, use_rff=False, kernel="gaussian", learn_kernel="off", learn_init_emb=False, msc_enable=False, msc_objective="legacy", msc_lambda=0.1, msc_beta=0.7, msc_tau=0.1, msc_k_min=8, msc_k_max=64, msc_n_anchors=4, msc_proj_dim=128, msc_min_anchor_frac=0.1, msc_detach_z=True, msc_view="subset", msc_focal_gamma=0.0, msc_anchor_power=1.0, msc_learn_gains=True, msc_pair_gap=0, msc_update_mode="joint", **kwargs):
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

        # Learnable initial-memory prior: m_t = (init_emb + sum_i E(x_i)) / (w + t),
        # w = exp(log_init_weight), init 0 -> w=1. False -> m_t = (sum_i E(x_i)) / t.
        self.learn_init_emb = learn_init_emb
        if self.learn_init_emb:
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
            self.ema_embedder if self.alternating_msc else self.embedder
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

        if self.learn_init_emb:
            info["init_emb_norm"] = self.init_emb.detach().norm()
            info["init_weight"] = self.log_init_weight.detach().exp()

        return output, (h_n, count_n), info

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
            h_0 = self.init_emb.view(1, 1, -1).expand(1, batch_size, -1)
            t_0 = self.log_init_weight.exp().view(1, 1, 1).expand(1, batch_size, 1)
        else:
            h_0 = ptu.zeros((1, batch_size, self.hidden_size))
            t_0 = ptu.zeros((1, batch_size, 1))
        return h_0, t_0

    def internal_state_to_hidden(self, internal_state):
        # Mirrors the forward output: running mean (⊙ MSC gains).
        hidden, count = internal_state
        out = hidden / count.clamp(min=1e-6)
        if self.msc is not None and self.msc_objective == "legacy":
            out = self.msc.gains() * out
        return out
