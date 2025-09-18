import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from .gpt2_vanilla import SinePositionalEncoding
from torchkit.networks import Mlp
import math


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, out_act = "linear", **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.transition_dropout_mask = None
        self.is_target = False
        self.embedder = Mlp(hidden_sizes=[4*hidden_size]*n_layer, output_size=hidden_size, input_size=input_size, 
                            norm = "layer", output_activation= out_act, dropout = 0.1, dropout_mode="all")

        self.temb_mode = kwargs["temb_mode"]
        assert self.temb_mode in ["none", "add", "concat"]
        print(f"Use Hist with temb_mode = {self.temb_mode}.")
        if self.temb_mode == "add":
            self.embed_timestep = SinePositionalEncoding(max_seq_length+1, hidden_size)
        elif self.temb_mode == "concat":
            self.embed_timestep = SinePositionalEncoding(max_seq_length+1, self.hidden_size // 2) # temb dimension is half of hidden_size
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
        mask = self.transition_dropout_mask
        if mask is None: # inference or no dropout
            mask = ptu.ones(L)
        if self.is_target: # for calculating target. Use current transitions with dropedout previous transitions
            t_expanded = (t + 1 + torch.cat((ptu.zeros(1), mask[:-1]), dim = 0).cumsum(dim = 0)).long() # (L,)
        else:
            t_expanded = (t + mask.cumsum(dim = 0)).long() # (L,)
        if self.temb_mode != "none":
            pe = self.embed_timestep(t_expanded).reshape(L, 1, -1) # t_expanded starts from 1
        
        if self.is_target:
            z = self.embedder(inputs) # (L, B, hidden_size)
            z_orig = z.clone()
            z = z * mask.reshape(-1, 1, 1) # (L, B, hidden_size)
            cumsum = hidden * t + z_orig + torch.cat((ptu.zeros(1, *z.shape[1:]), z[:-1]), dim = 0).cumsum(dim=0) # (L, B, hidden_size)
        else:
            z_partial = self.embedder(inputs[mask.bool()]) 
            z = ptu.zeros(L, *z_partial.shape[1:]) # (L, B, hidden_size)
            z[mask.bool()] = z_partial
            cumsum = hidden * t + z.cumsum(dim = 0) # (L, B, hidden_size)
        output = cumsum / t_expanded.clamp(min = 1).unsqueeze(-1).unsqueeze(-1) # when t = 0, output = 0
        h_n = output[-1].clone().unsqueeze(0), t_expanded[-1]
        if self.temb_mode == "add":
            output += pe
        if self.temb_mode == "concat":
            bs = output.shape[1]
            output = torch.cat((output, pe.repeat(1, bs, 1)), dim = -1)

        return output, h_n

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        h_0 = ptu.zeros((1, batch_size, self.hidden_size)).float()
        return h_0, 0 # (h_t, t)


    def sample_transition_dropout_mask(self, length, p):
        k = math.ceil((1 - p) * length)
        idx = torch.randperm(length)[:k]
        mask = ptu.zeros(length)
        mask[idx] = 1
        return mask


    def internal_state_to_hidden(self, internal_state):
        hidden, t = internal_state
        t_emb = self.embed_timestep(t).reshape(1, 1, -1) if self.temb_mode != "none" else 0
        if self.temb_mode == "add":
            hidden += t_emb
        if self.temb_mode == "concat":
            hidden = torch.cat((hidden, t_emb.repeat(1, hidden.shape[1], 1)), dim = -1)
        return hidden
