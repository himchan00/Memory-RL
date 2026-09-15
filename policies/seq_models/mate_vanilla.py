from copy import deepcopy

import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import InputNorm
from policies.seq_models.msc_aux import MSCAux
from policies.seq_models.msc_v2_aux import MSCV2Aux


class Mate(nn.Module):
    name = "mate"
    # A gate that reaches exactly 0 freezes the memory at the init prior AND
    # saturates the sigmoid, so no gradient can reopen it -- a dead end the
    # sparsity penalty will happily walk into. Rescaling sigmoid into
    # [_GATE_MIN, 1 - _GATE_MIN] keeps both directions reachable.
    _GATE_MIN = 0.01

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, dropout_ff=0.05, dropout_emb=0.05, use_gate=False, gate_sparsity_weight=0.0, gate_sparsity_target=0.06, learn_init_emb=False, use_ema_init_emb=False, ema_init_emb_beta=5e-4, use_store=False, store_grad_correction=True, msc_enable=False, msc_objective="legacy", msc_lambda=0.1, msc_beta=0.7, msc_tau=0.1, msc_k_min=8, msc_k_max=64, msc_n_anchors=4, msc_proj_dim=128, msc_min_anchor_frac=0.1, msc_detach_z=True, msc_view="subset", msc_focal_gamma=0.0, msc_anchor_power=1.0, msc_learn_gains=True, msc_pair_gap=0, msc_update_mode="joint", normalize_z=False, **kwargs):
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

        # One input projection followed by n_layer additional hidden blocks.
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout_emb),
        ]
        for _ in range(n_layer):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_ff),
            ]
        self.embedder = nn.Sequential(*layers)

        print(
            f"Mate embedder: n_layer={n_layer}, input_size={input_size}, "
            f"hidden_size={hidden_size}"
        )

        # Optional InputNorm on the transition embeddings before aggregation;
        # everything downstream (running mean, init_emb prior, MSC, z cache)
        # then lives in this normalized space.
        self.z_norm = InputNorm(hidden_size) if normalize_z else None

        # Initial-memory prior: m_t = (w * init_emb + sum_i E(x_i)) / (w + t),
        # where init_emb is learned or tracked as an EMA and w is always learned.
        # --- per-transition gate -------------------------------------------
        # MATE's memory is a UNIFORM mean, so the ~12 transitions that reveal
        # the frame map arrive at weight 1/200 each while ~188 uninformative
        # ones hold the rest. T-Maze survives far worse dilution (1 informative
        # step in 1001) because its corridor is DETERMINISTIC: the other
        # transitions contribute the same vector every episode, a removable
        # constant rather than interference. Alchemy's depend on which stone and
        # potion the policy happened to pick, so the signal is buried in
        # variance, and only a non-uniform weighting can dig it out.
        #
        # The gate weights the NUMERATOR and the DENOMINATOR alike, so the
        # memory stays a (weighted) mean -- bounded, and still invariant to the
        # ORDER of the transitions, which is what makes a mean the right
        # inductive bias for a CMDP. A transition gated to 0 leaves the memory
        # untouched instead of diluting it.
        #
        # The gate reads the raw transition, so it can separate an uninformative
        # no-op from an informative one: a stone that a potion failed to move
        # says that edge is BLOCKED, and must not be filtered away. That is also
        # why the gate is not keyed on ||delta_obs||, which would discard
        # exactly those.
        #
        # `gate_sparsity_weight` pushes mean(w) toward `gate_sparsity_target`.
        # Left free, the gate has no reason to shut anything -- the RL loss is
        # happy with a slightly reweighted average. The prior says only a small
        # fraction of transitions carry the chemistry (12/200 = 0.06 here), and
        # states it as a soft constraint rather than hoping it is discovered.
        self.use_gate = use_gate
        self.gate_sparsity_weight = float(gate_sparsity_weight)
        self.gate_sparsity_target = float(gate_sparsity_target)
        if self.use_gate:
            if not 0.0 < self.gate_sparsity_target <= 1.0:
                raise ValueError("gate_sparsity_target must be in (0, 1]")
            self.gate = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.LeakyReLU(),
                nn.Linear(hidden_size, 1), nn.Sigmoid(),
            )

        self.learn_init_emb = learn_init_emb
        self.use_ema_init_emb = use_ema_init_emb
        self.ema_init_emb_beta = float(ema_init_emb_beta)
        # STORE (Subset Training Over REused Embeddings): recompute only the
        # sampled transitions and reuse cached embeddings for the rest.
        self.use_store = bool(use_store)
        self.store_grad_correction = bool(store_grad_correction)
        if self.use_store and msc_enable:
            raise ValueError("use_store (STORE) is not supported with MSC")
        if self.use_gate and msc_enable:
            # MSCV2Aux rebuilds each subset's memory as (init + sum z)/count,
            # unweighted. Against a gated memory that is a different quantity,
            # and the contrastive loss would train a memory the policy never
            # sees. Combining them needs the weights plumbed into MSC first.
            raise ValueError("use_gate is not supported with MSC")
        if self.use_store and self.use_gate:
            # forward_cached reconstructs the count as init + physical_steps.
            # Under a gate the denominator is the cumulative WEIGHT, which the
            # cache does not carry, so the two are mutually exclusive.
            raise ValueError("use_gate is not supported with STORE")
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

    def embed_transitions(self, inputs, mask=None, update_norm=False):
        embedder = (
            self.ema_embedder if self.alternating_msc else self.embedder
        )
        return self._apply_z_norm(embedder(inputs), mask=mask, update=update_norm)

    def _apply_z_norm(self, z, mask=None, update=False):
        if self.z_norm is None:
            return z
        if update and self.training:
            self.z_norm.update_stats(z.detach(), mask=mask)
        return self.z_norm(z)

    def forward(
        self, inputs, h_0, mask=None, compute_msc=True,
        return_embeddings=False, **kwargs,
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
        # alternating_ema: stats follow the online embedder, updated in contrastive_loss()
        z = self.embed_transitions(
            inputs, mask=mask, update_norm=not self.alternating_msc,
        ) # (L, B, hidden_size)
        info = {}

        # cat([init, x]).cumsum(dim=0)[1:] == init + x.cumsum(dim=0)
        # avoids Inductor SplitScan + broadcast crash (pytorch/pytorch#180221)
        if self.use_gate:
            w = self._GATE_MIN + (1.0 - 2 * self._GATE_MIN) * self.gate(inputs)
            cumsum = torch.cat([hidden, z * w], dim=0).cumsum(dim=0)[1:]
            # The denominator accumulates the WEIGHTS, not the step count, so a
            # transition gated to 0 leaves the memory untouched rather than
            # diluting it. The output is still a mean, so it stays bounded.
            counts = initial_count + torch.cat(
                [torch.zeros_like(w[:1]), w], dim=0
            ).cumsum(dim=0)[1:]
            info["gate_mean"] = w.detach().squeeze(-1).mean(dim=1)
            info["gate_std"] = w.detach().squeeze(-1).std(dim=1)
            if self.training and self.gate_sparsity_weight > 0.0:
                if mask is None:
                    occupancy = w.mean()
                else:
                    m = mask.to(w.dtype)
                    occupancy = (w * m).sum() / m.sum().clamp(min=1.0)
                # Binary KL(occupancy || target), the sparse-autoencoder
                # penalty. Unlike a squared error it diverges as occupancy -> 0,
                # so shutting every transition is not a free minimum.
                q = occupancy.clamp(1e-6, 1.0 - 1e-6)
                p_t = self.gate_sparsity_target
                penalty = (
                    q * torch.log(q / p_t)
                    + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p_t))
                )
                info["_aux_loss"] = self.gate_sparsity_weight * penalty
                info["gate_occupancy"] = occupancy.detach()
        else:
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
        if return_embeddings:
            info["_transition_embeddings"] = z.detach()

        return output, (h_n, count_n), info

    def forward_cached(
        self,
        inputs,
        h_0,
        cached_embeddings,
        cached_prefixes,
        transition_t,
        mask=None,
    ):
        hidden, initial_count = h_0
        # cached_embeddings were stored post-normalization during rollout
        z = self.embed_transitions(inputs, mask=mask, update_norm=True)
        cached_embeddings = cached_embeddings.to(z)
        cached_prefixes = cached_prefixes.to(z)

        delta = z - cached_embeddings
        correction_before = torch.cat(
            (torch.zeros_like(delta[:1]), delta.cumsum(dim=0)[:-1]),
            dim=0,
        )
        # correction_before is the ONLY gradient path from the loss to z (next_joint
        # is consumed under no_grad in both agents), so every surviving pair has
        # i < t and needs BOTH rows sampled: p = k(k-1)/(T(T-1)). Paths that skip z
        # need one row: p = k/T. Under the shared 1/num_valid loss normalization the
        # embedder is therefore scaled down by (k-1)/(T-1); undo it with a
        # straight-through factor that leaves the forward value untouched.
        # Sound in window mode too: that path is reachable only without truncation,
        # where k == T and alpha == 1.
        if self.store_grad_correction and z.shape[0] > 1:
            alpha = (self.max_seq_length - 2) / (z.shape[0] - 1)  # (T-1)/(k-1)
            frozen = correction_before.detach()
            correction_before = frozen + alpha * (correction_before - frozen)
        current_sums = hidden + cached_prefixes + correction_before
        next_sums = current_sums + z

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
        if self.z_norm is not None:
            info["z_norm_mu"] = self.z_norm.mu.mean()
            info["z_norm_sigma"] = self.z_norm.sigma.mean()
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
        z = self._apply_z_norm(self.embedder(inputs), mask=mask, update=True)

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
