import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from .gpt2_vanilla import SinePositionalEncoding
import math


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, max_seq_length, agg = "sum", out_act = "linear", transition_dropout = 0, **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        self.out_activation = get_activation(out_act)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.transition_dropout = transition_dropout

        assert agg in ["sum", "logsumexp", "mean"]
        self.agg = agg
        if self.agg == "mean":
            self.temb_mode = kwargs["temb_mode"]
            assert self.temb_mode in ["none", "add", "concat"]
            if self.temb_mode == "add":
                self.embed_timestep = SinePositionalEncoding(max_seq_length+1, hidden_size)
            elif self.temb_mode == "concat":
                self.embed_timestep = SinePositionalEncoding(max_seq_length+1, kwargs["temb_size"])
            else:
                self.embed_timestep = None

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size) except for agg = "mean, where h_0: (1, B, hidden_size), int 
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size) except for agg = "mean, where h_n: (1, B, hidden_size), int 
        """
        if self.agg == "sum":
            z = self.out_activation(inputs)
            z = torch.cat((h_0 * (self.max_seq_length ** 0.5), z), dim = 0)
            output = torch.cumsum(z, dim = 0)[1:] / (self.max_seq_length ** 0.5)
            h_n = output[-1].unsqueeze(0)
        elif self.agg == "logsumexp":
            z = self.out_activation(inputs)
            z = torch.cat((h_0, z), dim = 0)
            max_z, _ = torch.cummax(z, dim = 0)
            output = (max_z + torch.logcumsumexp(z - max_z, dim = 0))[1:] # For numerical stability
            h_n = output[-1].unsqueeze(0)
        elif self.agg == "mean":
            L = inputs.shape[0]
            (hidden, t) = h_0
            mask = self.transition_dropout_mask(L) # (L,)
            t_expanded = ptu.arange(t+1, t+L+1) # (L,)
            t_denom = (t + mask.cumsum(dim = 0)).long() # (L,)
            if self.temb_mode != "none":
                pe = self.embed_timestep(t_expanded).reshape(L, 1, -1) # t_expanded starts from 1
            z = self.out_activation(inputs)
            z = z * mask.reshape(-1, 1, 1) # (L, bs, hidden_size)
            cumsum = (hidden * t + z.cumsum(dim = 0)) # (L, bs, hidden_size)
            output = cumsum / t_denom.clamp(min=1).unsqueeze(-1).unsqueeze(-1) # when t = 0, output = 0
            if self.temb_mode == "add":
                output = output + pe
            h_n = output[-1].unsqueeze(0), t_expanded[-1]
            if self.temb_mode == "concat":
                bs = output.shape[1]
                output = torch.cat((output, pe.repeat(1, bs, 1)), dim = -1)

        return output, h_n

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        h_0 = ptu.zeros((1, batch_size, self.hidden_size)).float()
        if self.agg == "mean":
            return h_0, 0 # (h_t, t)
        else:
            return h_0 # (h_t)

    def transition_dropout_mask(self, length):
        if self.training and self.transition_dropout > 0:
            p = torch.empty(1).uniform_(0, self.transition_dropout).item()
            k = math.ceil((1 - p) * length)
            idx = torch.randperm(length)[:k]
            mask = ptu.zeros(length)
            mask[idx] = 1
            return mask
        else:
            return ptu.ones(length)

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