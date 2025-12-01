import torch.nn as nn
import torchkit.pytorch_utils as ptu


class Markov(nn.Module):
    name = "markov"

    def __init__(self, hidden_size=0, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        return
        output: (T, B, 0)
        """
        output = inputs.new_zeros(inputs.size(0), inputs.size(1), 0)
        return output, h_0

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        return ptu.zeros((batch_size, 0)).float()