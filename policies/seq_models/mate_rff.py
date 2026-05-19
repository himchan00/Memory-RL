import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu


class MateRff(nn.Module):
    name = "mate_rff"

    def __init__(self, input_size, hidden_size, max_seq_length, init_emb_zero=False, **kwargs):
        super().__init__()
        assert input_size == hidden_size, (
            f"MateRff expects RFF-embedded inputs with input_size == hidden_size "
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length

        if init_emb_zero:
            self.register_buffer("init_emb", ptu.zeros(self.hidden_size))
        else:
            self.init_emb = nn.Parameter(ptu.randn(self.hidden_size))

    def forward(self, inputs, h_0, **kwargs):
        """
        inputs: (T, B, hidden_size)  — already RFF-embedded transitions
        h_0:    ((1, B, hidden_size), (1, B, 1))
        return:
            output: (T, B, hidden_size)  — running mean memory at each step
            h_n:    ((1, B, hidden_size), (1, B, 1))
            info:   {}
        """
        hidden, t = h_0
        z = inputs
        w = inputs.new_ones((inputs.shape[0], inputs.shape[1], 1))

        # cat([init, x]).cumsum(dim=0)[1:] == init + x.cumsum(dim=0)
        # avoids Inductor SplitScan + broadcast crash (pytorch/pytorch#180221)
        cumsum = torch.cat([hidden, z * w], dim=0).cumsum(dim=0)[1:]
        t_expanded = torch.cat([t, w], dim=0).cumsum(dim=0)[1:]
        h_n = cumsum[-1].clone().unsqueeze(0)
        t_n = t_expanded[-1].clone().unsqueeze(0)
        output = cumsum / t_expanded.clamp(min=1e-6)
        return output, (h_n, t_n), {}

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        h_0 = self.init_emb.unsqueeze(0).expand(1, batch_size, -1)  # (1, B, hidden_size)
        t_0 = ptu.ones((1, batch_size, 1))  # count init_emb as 1 transition embedding
        return h_0, t_0

    def internal_state_to_hidden(self, internal_state):
        return internal_state[0]
