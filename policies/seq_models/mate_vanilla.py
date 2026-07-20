import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp
from policies.seq_models.Rff_embedding import RFFEmbedding
from policies.seq_models.gpt2_vanilla import SinePositionalEncoding
from policies.seq_models.msc_aux import MSCAux


class Mate(nn.Module):
    name = "mate"
    _GATE_MIN = 0.01  # gate floor/ceiling/collapse threshold

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, dropout_ff=0.05, dropout_emb=0.05, use_gate=False, gate_noise_std=0.0, transition_dropout=0.0, rollout_dropout=0.0, use_rff=False, kernel="gaussian", learn_kernel="off", add_positional_embedding=True, learnable_pe_scale=False, learn_init_emb=False, msc_enable=False, msc_lambda=0.1, msc_beta=0.7, msc_tau=0.1, msc_n_anchors=4, msc_proj_dim=128, msc_min_anchor_frac=0.1, msc_detach_z=True, msc_view="subset", msc_focal_gamma=0.0, msc_anchor_power=1.0, **kwargs):
        super().__init__()
        # input_size = raw transition_size (post-InputNorm); RNN_head sets transition_embedder=Identity for mate.
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.use_gate = use_gate
        self.gate_noise_std = gate_noise_std

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

        # Gate network shape (input_size, hidden_size, 1).
        if self.use_gate:
            print("Using gate in Mate")
            self.gate = Mlp(input_size=input_size, output_size=1, hidden_sizes=[hidden_size], output_activation='linear', dropout=dropout_ff)

        # Learnable initial-memory prior: m_t = (init_emb + sum_i E(x_i)) / (w + t),
        # w = exp(log_init_weight), init 0 -> w=1. False -> m_t = (sum_i E(x_i)) / t.
        self.learn_init_emb = learn_init_emb
        if self.learn_init_emb:
            self.init_emb = nn.Parameter(ptu.randn(self.hidden_size))
            self.log_init_weight = nn.Parameter(ptu.zeros(()))

        # MSC contrastive aux (see msc_aux.py). None → every msc hook below
        # is a no-op and the baseline is reproduced exactly.
        self.msc = MSCAux(
            hidden_size, msc_lambda=msc_lambda, beta=msc_beta, tau=msc_tau,
            n_anchors=msc_n_anchors, proj_dim=msc_proj_dim,
            min_anchor_frac=msc_min_anchor_frac, detach_inputs=msc_detach_z,
            view=msc_view, focal_gamma=msc_focal_gamma, anchor_power=msc_anchor_power,
        ) if msc_enable else None
        if self.msc is not None:
            print(f"Using MSC in Mate: lambda={msc_lambda}, beta={msc_beta}, tau={msc_tau}, n_anchors={msc_n_anchors}, proj_dim={msc_proj_dim}, detach_z={msc_detach_z}, view={msc_view}, focal_gamma={msc_focal_gamma}, anchor_power={msc_anchor_power}")

        self.transition_dropout = float(transition_dropout)
        self.rollout_dropout = float(rollout_dropout)
        assert 0.0 <= self.transition_dropout < 1.0, "transition_dropout must be in [0, 1)"
        assert 0.0 <= self.rollout_dropout < 1.0, "rollout_dropout must be in [0, 1)"
        self._rollout_dropout_active = True

        # Fixed sinusoidal PE added to the running-mean output;
        self.add_positional_embedding = add_positional_embedding
        self.learnable_pe_scale = learnable_pe_scale and add_positional_embedding
        if add_positional_embedding:
            print(f"Using positional embedding in Mate: learnable_pe_scale={self.learnable_pe_scale}")
            self.pe = SinePositionalEncoding(max_seq_length, hidden_size)
            if self.learnable_pe_scale:
                # log-parameterized scalar; init 0 → scale=1 (baseline)
                self.log_pe_scale = nn.Parameter(torch.zeros(()))

    def forward(self, inputs, h_0, mask=None, **kwargs):
        """
        inputs: (T, B, input_size)
        h_0: (1, B, hidden_size), (1, B, 1), (1, B, 1)   # hidden, gated count t, integer step counter pos
        mask: optional (T, B, 1) validity mask (training only; consumed by MSC anchor sampling)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), (1, B, 1), (1, B, 1)
        """
        hidden, t, pos = h_0
        z = self.embedder(inputs) # (L, B, hidden_size)
        if self.use_gate:
            logits = self.gate(inputs) # (T, B, 1)

            # Noisy gating: inject noise only when gradients are enabled
            if torch.is_grad_enabled() and self.gate_noise_std > 0.0:
                logits = logits + torch.randn_like(logits) * self.gate_noise_std

            # Sigmoid + affine rescaling to [_GATE_MIN, 1 - _GATE_MIN]
            raw_w = torch.sigmoid(logits)
            w = self._GATE_MIN + (1.0 - 2 * self._GATE_MIN) * raw_w

            info = {
                "gates_mean": w.detach().squeeze(-1).mean(dim=1),
                "gates_std": w.detach().squeeze(-1).std(dim=1),
                "gates_collapse_ratio": (raw_w.detach() < self._GATE_MIN).float().mean(),
            }
        else:
            w = inputs.new_ones((inputs.shape[0], inputs.shape[1], 1))
            info = {}

        if self.training:
            drop_prob = self.transition_dropout
        else:
            drop_prob = self.rollout_dropout if self._rollout_dropout_active else 0.0

        if drop_prob > 0.0:
            keep = torch.bernoulli(torch.full_like(w, 1.0 - drop_prob))
            z_dropped = z * keep
            w_dropped = w * keep
            info["transition_keep_rate"] = keep.detach().mean()
            info["transition_keep_std"] = keep.detach().std()
        else:
            keep = None
            z_dropped = z
            w_dropped = w
        
            
        # cat([init, x]).cumsum(dim=0)[1:] == init + x.cumsum(dim=0)
        # avoids Inductor SplitScan + broadcast crash (pytorch/pytorch#180221)
        cumsum = torch.cat([hidden, z_dropped * w_dropped], dim=0).cumsum(dim=0)[1:]
        t_expanded = torch.cat([t, w_dropped], dim=0).cumsum(dim=0)[1:] # (T, B, 1)
        h_n = cumsum[-1].clone().unsqueeze(0)
        t_n = t_expanded[-1].clone().unsqueeze(0)
        output = cumsum / t_expanded.clamp(min=1e-6) # (L, B, hidden_size)

        if self.training and self.transition_dropout > 0.0:
            correction = 1.0 - keep
            cumsum_aligned = cumsum + z* w* correction
            t_expanded_aligned = t_expanded + w * correction
            output_target = cumsum_aligned / t_expanded_aligned.clamp(min=1e-6)
        else:
            output_target = None

        # MSC (see msc_aux.py): InfoNCE over sub-multiset views (training only),
        # then spectral gains s on the policy path. Gains are applied pre-PE;
        # by linearity s ⊙ m_t equals the running mean of s ⊙ z — a diagonal
        # reweighting of the kernel's spectral measure (Prop 5), so the memory
        # stays a kernel mean embedding under the reweighted kernel.
        if self.msc is not None:
            if self.training:
                msc_loss, msc_info = self.msc(
                    z_contrib=z_dropped * w_dropped, init_hidden=hidden,
                    cumsum=cumsum, t_expanded=t_expanded, mask=mask,
                )
                info["_aux_loss"] = msc_loss
                info.update(msc_info)
            gains = self.msc.gains()
            output = gains * output
            if output_target is not None:
                output_target = gains * output_target

        # Add fixed sinusoidal PE keyed by absolute step index. arange(T) on
        # the same device as `pos` keeps this inside compiled graphs.
        T = inputs.shape[0]
        if self.add_positional_embedding:
            pos_idx = (pos + torch.arange(T, device=pos.device).view(T, 1, 1)).squeeze(-1)
            pe = self.pe(pos_idx)  # (T, B, hidden_size)
            if self.learnable_pe_scale:
                pe = self.log_pe_scale.exp() * pe
            output = output + pe
            if output_target is not None:
                output_target = output_target + pe
        pos_n = pos + T

        if output_target is not None:
            info["_output_target"] = output_target

        if self._rff_layer is not None:
            info.update(self._rff_layer.logging_stats())

        if self.learn_init_emb:
            info["init_emb_norm"] = self.init_emb.detach().norm()
            info["init_weight"] = self.log_init_weight.detach().exp()

        if self.learnable_pe_scale:
            info["pe_scale"] = self.log_pe_scale.detach().exp()

        return output, (h_n, t_n, pos_n), info

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        """internal state: (hidden_state, gated count t, integer step counter pos)"""
        if self.learn_init_emb:
            h_0 = self.init_emb.view(1, 1, -1).expand(1, batch_size, -1)
            t_0 = self.log_init_weight.exp().view(1, 1, 1).expand(1, batch_size, 1)
        else:
            h_0 = ptu.zeros((1, batch_size, self.hidden_size))
            t_0 = ptu.zeros((1, batch_size, 1))
        pos_0 = ptu.zeros((1, batch_size, 1)).long() # next step's absolute position
        return h_0, t_0, pos_0

    def internal_state_to_hidden(self, internal_state):
        # Mirrors the forward output: running mean (⊙ MSC gains) + sinusoidal PE at `pos`.
        hidden, t, pos = internal_state
        out = hidden / t.clamp(min=1e-6)
        if self.msc is not None:
            out = self.msc.gains() * out
        if self.add_positional_embedding:
            pe = self.pe(pos.squeeze(-1))
            if self.learnable_pe_scale:
                pe = self.log_pe_scale.exp() * pe
            out = out + pe
        return out
