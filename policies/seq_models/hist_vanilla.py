import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from .gpt2_vanilla import SinePositionalEncoding


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, max_seq_length, out_act = "linear", **kwargs):
        super().__init__()
        self.out_activation = get_activation(out_act)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.permutation_idx = None
        self.is_target = False

        self.temb_mode = kwargs["temb_mode"]
        assert self.temb_mode in ["none", "add", "concat"]
        print(f"Use Hist with temb_mode = {self.temb_mode}.")
        if self.temb_mode == "add":
            self.embed_timestep = SinePositionalEncoding(max_seq_length+1, hidden_size)
        elif self.temb_mode == "concat":
            self.embed_timestep = SinePositionalEncoding(max_seq_length+1, kwargs["temb_size"])
        else:
            self.embed_timestep = None

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size), int 
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), int 
        """
        L = inputs.shape[0]
        (hidden, t) = h_0
        t_expanded = ptu.arange(t+1, t+L+1) # (L,)
        if self.temb_mode != "none":
            pe = self.embed_timestep(t_expanded).reshape(L, 1, -1) # t_expanded starts from 1
        if self.temb_mode == "add":
            hidden -= self.embed_timestep(t).reshape(1, 1, -1)
        z = self.out_activation(inputs) # (L, B, hidden_size)
        if self.permutation_idx is not None:
            if self.is_target:
                z_orig = z.clone()
            z = z[self.permutation_idx]
        cumsum = (hidden * t + z.cumsum(dim = 0)) # (L, B, hidden_size)
        if self.is_target:
            cumsum = z_orig + torch.cat((torch.zeros(1, *cumsum.shape[1:]).to(cumsum.device), cumsum[:-1]), dim = 0) # (L, B, hidden_size)
        output = cumsum / t_expanded.unsqueeze(-1).unsqueeze(-1) # when t = 0, output = 0
        if self.temb_mode == "add":
            output += pe
        h_n = output[-1].unsqueeze(0), t_expanded[-1]
        if self.temb_mode == "concat":
            bs = output.shape[1]
            output = torch.cat((output, pe.repeat(1, bs, 1)), dim = -1)

        return output, h_n

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        h_0 = ptu.zeros((1, batch_size, self.hidden_size)).float()
        if self.temb_mode == "add":
            h_0 += self.embed_timestep(0).reshape(1, 1, -1)
        return h_0, 0 # (h_t, t)


def get_activation(s_act):
    if s_act == 'relu':
        return nn.ReLU(inplace=True)
    elif s_act == 'sigmoid':
        return nn.Sigmoid()
    elif s_act == 'softplus':
        return nn.Softplus()
    elif s_act == 'linear':
        return nn.Identity()
    elif s_act == 'tanh':
        return nn.Tanh()
    elif s_act == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')