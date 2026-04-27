import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu


_NEG_INF = -1e30  # finite stand-in for -inf so initial cummax doesn't poison gradients


class SplAgger(nn.Module):
    """SplAgger: GRU + causal split-aggregator (Beck et al. 2024, arXiv:2403.03020).

    The RNN output (T, B, hidden_size) is split in half along the channel dim;
    one half is left as-is (permutation-variant), the other is causally
    aggregated across time (permutation-invariant). The two halves are then
    concatenated back.
    """
    name = "splagger"

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layer,
        pdrop=0.1,
        agg_type="max",
        rnn_type="gru",
        **kwargs,
    ):
        super().__init__()
        assert hidden_size % 2 == 0, "splagger requires even hidden_size (chunk in half)"
        assert agg_type in ("max", "mean"), f"unsupported agg_type: {agg_type}"
        assert rnn_type == "gru", "only GRU base RNN supported in V1 (matches paper)"

        self.hidden_size = hidden_size
        self.half = hidden_size // 2
        self.num_layers = n_layer
        self.agg_type = agg_type

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=False,
            dropout=pdrop,
            bias=True,
        )
        self._initialize_rnn()

    def _initialize_rnn(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, inputs, h_0, **kwargs):
        """
        inputs: (T, B, input_size)
        h_0:    (gru_hidden, agg_state, t_count) — see get_zero_internal_state
        Returns: output (T, B, hidden_size), h_n (same layout), info dict
        """
        gru_h0, agg_state, t_count = h_0

        out, gru_hn = self.rnn(inputs, gru_h0)
        skip, agg_in = torch.chunk(out, 2, dim=-1)

        # cat(prev, x).cum*(dim=0)[1:] — same idiom as mate_vanilla.py:62, makes
        # forward()/step() numerically equal and avoids the Inductor SplitScan crash.
        if self.agg_type == "max":
            stacked = torch.cat([agg_state, agg_in], dim=0)
            cum = stacked.cummax(dim=0).values[1:]
            new_agg_state = cum[-1:].clone()
            new_t_count = t_count
            aggregated = cum
        else:  # mean
            cum_sum = torch.cat([agg_state, agg_in], dim=0).cumsum(dim=0)[1:]
            T = agg_in.shape[0]
            step_counts = torch.arange(1, T + 1, device=agg_in.device, dtype=agg_in.dtype)
            t_expanded = t_count + step_counts.view(-1, 1, 1)
            aggregated = cum_sum / t_expanded.clamp(min=1.0)
            new_agg_state = cum_sum[-1:].clone()
            new_t_count = t_expanded[-1:].clone()

        output = torch.cat([skip, aggregated], dim=-1)
        return output, (gru_hn, new_agg_state, new_t_count), {}

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        gru_h = ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()
        if self.agg_type == "max":
            agg = ptu.zeros((1, batch_size, self.half)).float() + _NEG_INF
        else:
            agg = ptu.zeros((1, batch_size, self.half)).float()
        t = ptu.zeros((1, batch_size, 1)).float()
        return gru_h, agg, t

    def internal_state_to_hidden(self, internal_state):
        gru_h, _, _ = internal_state
        return gru_h[-1].unsqueeze(0)
