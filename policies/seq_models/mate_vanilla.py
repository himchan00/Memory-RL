import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp, gpt_like_Mlp


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, pdrop, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedder = gpt_like_Mlp(hidden_size=hidden_size, n_layer=n_layer, pdrop=pdrop, use_output_ln=False)
        # self.embedder = Mlp(hidden_sizes=[hidden_size]*(n_layer-1), # one layer is used in transition embedder
        #                     output_size=hidden_size, input_size=input_size, output_activation= out_act, norm = norm, dropout = pdrop)
        self.init_emb = nn.Parameter(ptu.randn(self.hidden_size))

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size), (1, B, 1)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), (1, B, 1)
        """
        T = len(inputs)
        hidden, t = h_0
        z = self.embedder(inputs) # (L, B, hidden_size)
        cumsum = hidden + z.cumsum(dim=0)
        t_expanded = t + ptu.arange(1, T+1).view(T, 1, 1)
        h_n = cumsum[-1].clone().unsqueeze(0)
        t_n = t + T
        output = (cumsum + self.init_emb) / t_expanded
        return output, (h_n, t_n)

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        """
        internal state: (hidden_state, time_step)
        )
        """
        return ptu.zeros((1, batch_size, self.hidden_size)).float(), ptu.zeros((1, batch_size, 1))

    def internal_state_to_hidden(self, internal_state):
        return internal_state[0]  # first element is hidden state for Mate
