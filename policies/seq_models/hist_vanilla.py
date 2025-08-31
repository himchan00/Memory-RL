import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp
from .gpt2_vanilla import SinePositionalEncoding


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, agg = "sum", out_act = "linear", pdrop = 0.1, norm = "none", **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        self.encoder = Mlp(hidden_sizes=[4*hidden_size]*n_layer, output_size=hidden_size, 
                           input_size = input_size, output_activation=get_activation(out_act), dropout=pdrop, norm = norm, norm_final=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        assert agg in ["sum", "logsumexp", "mean", "gaussian"]
        self.agg = agg
        if self.agg == "gaussian":
            assert hidden_size % 2 == 0, "hidden_size must be even when agg = gaussian"
        if self.agg == "mean":
            self.temb_mode = kwargs["temb_mode"]
            assert self.temb_mode in ["none", "input", "output", "concat"]
            if self.temb_mode == "input":
                self.embed_timestep = SinePositionalEncoding(max_seq_length, input_size)
            elif self.temb_mode == "output":
                self.embed_timestep = SinePositionalEncoding(max_seq_length, hidden_size)
            elif self.temb_mode == "concat":
                self.embed_timestep = SinePositionalEncoding(max_seq_length, kwargs["temb_size"])
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
            z = self.encoder(inputs)
            z = torch.cat((h_0 * (self.max_seq_length ** 0.5), z), dim = 0)
            output = torch.cumsum(z, dim = 0)[1:] / (self.max_seq_length ** 0.5)
            h_n = output[-1].unsqueeze(0)
        elif self.agg == "gaussian":
            mu, log_unscaled_prec = self.encoder(inputs).chunk(2, dim=-1)
            prec = torch.exp(log_unscaled_prec) / self.max_seq_length
            mu_times_prec = mu * prec
            prev_mu_times_prec, prev_prec = h_0.chunk(2, dim=-1)
            mu_times_prec = torch.cat((prev_mu_times_prec, mu_times_prec), dim=0)
            prec = torch.cat((prev_prec, prec), dim=0)
            new_mu_times_prec = torch.cumsum(mu_times_prec, dim=0)[1:]
            new_prec = torch.cumsum(prec, dim=0)[1:]
            output = torch.cat((new_mu_times_prec / new_prec, new_prec), dim = -1)
            h_n = torch.cat((new_mu_times_prec[-1], new_prec[-1]), dim=-1).unsqueeze(0)
        elif self.agg == "logsumexp":
            z = self.encoder(inputs)
            z = torch.cat((h_0, z), dim = 0)
            max_z, _ = torch.cummax(z, dim = 0)
            output = (max_z + torch.logcumsumexp(z - max_z, dim = 0))[1:] # For numerical stability
            h_n = output[-1].unsqueeze(0)
        elif self.agg == "mean":
            L = inputs.shape[0]
            (hidden, t) = h_0
            t_expanded = ptu.arange(t+1, t+L+1) # (L,)
            if self.temb_mode != "none":
                pe = self.embed_timestep(t_expanded).reshape(L, 1, -1)
            if self.temb_mode == "input":
                inputs = inputs + pe
            z = self.encoder(inputs)
            z = torch.cat((hidden * t, z), dim = 0)
            cumsum = torch.cumsum(z, dim = 0)[1:] # (L, bs, hidden_size)
            output = cumsum / t_expanded.unsqueeze(-1).unsqueeze(-1)
            if self.temb_mode == "output":
                output = output + pe
            h_n = output[-1].unsqueeze(0), t+L
            if self.temb_mode == "concat":
                bs = output.shape[1]
                output = torch.cat((output, pe.repeat(1, bs, 1)), dim = -1)

        return output, h_n

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        h_0 = ptu.zeros((1, batch_size, self.hidden_size)).float()
        if self.agg == "gaussian":
            h_0[:, :, self.hidden_size // 2:] = 1.0 # Init prec = 1
        if self.agg == "mean":
            if self.temb_mode == "output":
                h_0 = h_0 + self.embed_timestep(0).reshape(1, 1, -1)
            return h_0, 0 # (h_t, t)
        else:
            return h_0 # (h_t)
    
    def get_zero_hidden_state(self, batch_size=1):
        hidden = ptu.zeros((1, batch_size, self.hidden_size)).float()
        if self.agg == "gaussian":
            hidden[:, :, self.hidden_size // 2:] = 1.0 # Init prec = 1
        if self.agg == "mean":
            if self.temb_mode == "output":
                hidden = hidden + self.embed_timestep(0).reshape(1, 1, -1)
            if self.temb_mode == "concat":
                hidden = torch.concat((hidden, self.embed_timestep(0).reshape(1, 1, -1).repeat(1, batch_size, 1)), dim = -1)
        return hidden


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
        return nn.LeakyReLU(0.2, inplace=True)
    elif s_act == 'softmax':
        return nn.Softmax(dim=1)
    elif s_act == 'swish':
        return nn.SiLU(inplace=True)
    else:
        raise ValueError(f'Unexpected activation: {s_act}')