import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp, gpt_like_Mlp


class Mate(nn.Module):
    name = "mate"
    _GATE_MIN = 0.01  # gate floor/ceiling/collapse threshold

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, pdrop, use_gate=False, gate_noise_std=0.0, init_emb_zero=False, transition_dropout=0.0, rollout_dropout=0.0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.use_gate = use_gate
        self.gate_noise_std = gate_noise_std
        self.use_output_ln = kwargs.get("use_output_ln", True)
        print(f"use_output_ln={self.use_output_ln}")
        self.embedder = gpt_like_Mlp(hidden_size=hidden_size, n_layer=n_layer, pdrop=pdrop, use_output_ln=self.use_output_ln)
        if init_emb_zero:
            self.register_buffer("init_emb", ptu.zeros(self.hidden_size))
        else:
            self.init_emb = nn.Parameter(ptu.randn(self.hidden_size))
        if self.use_gate:
            print("Using gate in Mate")
            self.gate = Mlp(
                input_size=input_size, 
                output_size=1, 
                hidden_sizes=[hidden_size] * n_layer,
                output_activation='linear',
                dropout=pdrop
            )
        self.transition_dropout = float(transition_dropout)
        self.rollout_dropout = float(rollout_dropout)
        assert 0.0 <= self.transition_dropout < 1.0, "transition_dropout must be in [0, 1)"
        assert 0.0 <= self.rollout_dropout < 1.0, "rollout_dropout must be in [0, 1)"
        self._rollout_dropout_active = True

    def forward(self, inputs, h_0, **kwargs):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size), (1, B, 1)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), (1, B, 1)
        """
        hidden, t = h_0
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
            info["_output_target"] = output_target

        info["init_emb_norm"] = self.init_emb.detach().norm()

        return output, (h_n, t_n), info

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        """
        internal state: (hidden_state, time_step)
        )
        """
        h_0 = self.init_emb.unsqueeze(0).expand(1, batch_size, -1)  # (1, B, hidden_size)
        t_0 = ptu.ones((1, batch_size, 1)) # Count init_emb as 1 transtion embedding
        return h_0, t_0

    def internal_state_to_hidden(self, internal_state):
        return internal_state[0]  # first element is hidden state for Mate
