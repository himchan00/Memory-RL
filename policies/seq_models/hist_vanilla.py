import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
import math
from torchkit.networks import Mlp


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, agg = "sum", out_act = "linear", t_emb_size = 128, pdrop = 0.1, **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        self.encoder = Mlp(hidden_sizes=[4*hidden_size]*n_layer, output_size=hidden_size, 
                           input_size = input_size, output_activation=get_activation(out_act), dropout=pdrop)
        self.hidden_size = hidden_size
        assert agg in ["sum", "logsumexp", "mean", "mean_temb"]
        self.agg = agg
        self.t_emb_size = t_emb_size if self.agg == "mean_temb" else None # Only used when agg = "mean_temb"

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size) if self.agg = "sum else h_0: (1, B, hidden_size+1) Last hidden represents the timestep
        return
        outputs: (T, B, hidden_size) if self.agg = "sum" else outputs: (T, B, hidden_size + t_emb_dim) Last t_emb_dim represents the time embedding
        h_n: (1, B, hidden_size) if self.agg = "sum else h_n: (1, B, hidden_size+1) Last hidden represents the timestep
        """
        z = self.encoder(inputs)
        if self.agg == "sum":
            z = torch.cat((h_0, z), dim = 0)
            output = torch.cumsum(z, dim = 0)[1:]
            h_n = output[-1].unsqueeze(0)
        elif self.agg == "logsumexp":
            z = torch.cat((h_0, z), dim = 0)
            max_z, _ = torch.cummax(z, dim = 0)
            output = (max_z + torch.logcumsumexp(z - max_z, dim = 0))[1:] # For numerical stability
            h_n = output[-1].unsqueeze(0)
        else: # "mean" or "mean_temb"
            L = inputs.shape[0]
            hidden = h_0[:, :, :-1] # (1, bs, hidden_dim)
            t = h_0[:, :, -1].unsqueeze(-1) # (1, bs, 1)
            t_expanded = t + 1 + ptu.arange(0, L).view(L, 1, 1) # (L, bs, 1)
            z = torch.cat((hidden * t, z), dim = 0)
            cumsum = torch.cumsum(z, dim = 0)[1:] # (L, bs, hidden_size)
            output = cumsum / t_expanded
            if self.agg == "mean_temb":
                t_emb = get_timestep_embedding(t_expanded.flatten(), embedding_dim=self.t_emb_size).reshape(L, -1, self.t_emb_size) # (L, bs, t_emb_size)
                output = torch.cat((output, t_emb), dim = -1) # (L, bs, hidden_size + t_emb_size)
            h_n = torch.cat((output[-1, :, :self.hidden_size].unsqueeze(0), t+L), dim = -1) # (1, bs, hidden_size + 1)

        return output, h_n

    def get_zero_internal_state(self, batch_size=1):
        if (self.agg == "mean" or self.agg == "mean_temb"):
            return ptu.zeros((1, batch_size, self.hidden_size+1)).float() # (h_t, t)
        else:
            return ptu.zeros((1, batch_size, self.hidden_size)).float() # (h_t)




def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    emb = timesteps.type(dtype=torch.float)[:, None] * emb[None, :].to(timesteps.device)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.pad(emb, [0, 1], value=0.0)
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def process_single_t(x, t):
    """make single integer t into a vector of an appropriate size"""
    if isinstance(t, int) or len(t.shape) == 0 or len(t) == 1:
        t = torch.ones([x.shape[0]], dtype=torch.long, device=x.device) * t
    return t


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