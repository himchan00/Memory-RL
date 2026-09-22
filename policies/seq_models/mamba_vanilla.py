import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba as MambaMixer
except ImportError:
    MambaMixer = None


class Mamba(nn.Module):
    """AMAGO-style Mamba: n_layer x (x + Dropout(Mamba(LN(x)))) -> LN."""

    name = "mamba"

    def __init__(self, input_size, hidden_size, n_layer, dropout_ff=0.05,
                 d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        assert MambaMixer is not None, "mamba needs mamba_ssm + causal_conv1d (see CLAUDE.md)"
        assert input_size == hidden_size
        self.hidden_size = hidden_size
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(n_layer))
        self.mixers = nn.ModuleList(
            MambaMixer(hidden_size, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layer)
        )
        self.dropout = nn.Dropout(dropout_ff)
        self.norm_f = nn.LayerNorm(hidden_size)

    def forward(self, inputs, h_0, **kwargs):
        """
        inputs: (T, B, hidden_size)
        h_0: None (training, zero state) or per-layer (conv_state, ssm_state) for T=1 rollout
        """
        x = inputs.transpose(0, 1)  # (B, T, hidden_size)
        h_n = None if h_0 is None else []
        for i, (norm, mixer) in enumerate(zip(self.norms, self.mixers)):
            if h_0 is None:
                y = mixer(norm(x))
            else:  # states are updated in place
                y, conv_state, ssm_state = mixer.step(norm(x), *h_0[i])
                h_n.append((conv_state, ssm_state))
            x = x + self.dropout(y)
        return self.norm_f(x).transpose(0, 1), h_n

    def get_zero_internal_state(self, batch_size=1, training=False, **kwargs):
        if training:
            return None
        return [m.allocate_inference_cache(batch_size, max_seqlen=1) for m in self.mixers]
