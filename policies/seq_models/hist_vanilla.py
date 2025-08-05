import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
import math
from torch import Tensor


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, agg = "sum", out_act = "linear", t_emb_size = 128, pdrop = 0.1, layer_norm_epsilon = 1e-5, **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        assert input_size == hidden_size, "input size must be equal to hidden size for residual connection"
        l_layer = [Block(hidden_size=hidden_size, dropout=pdrop, layer_norm_epsilon=layer_norm_epsilon) for _ in range(n_layer)]
        self.ln_in = nn.LayerNorm(input_size, layer_norm_epsilon)
        self.encoder = nn.Sequential(*l_layer)
        self.out_act = get_activation(out_act)
        self.hidden_size = hidden_size
        self.num_layers = n_layer
        assert agg in ["sum", "mean"]
        self.agg = agg
        self.t_emb_size = t_emb_size if self.agg == "mean" else None # Only used when agg = "mean"

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size) if self.agg = "sum else h_0: (1, B, hidden_size+1) Last hidden represents the timestep
        return
        outputs: (T, B, hidden_size) if self.agg = "sum" else outputs: (T, B, hidden_size + t_emb_dim) Last t_emb_dim represents the time embedding
        h_n: (1, B, hidden_size) if self.agg = "sum else h_n: (1, B, hidden_size+1) Last hidden represents the timestep
        """
        z = self.encoder(self.ln_in(inputs))
        z = self.out_act(z)
        if self.agg == "mean":
            L = inputs.shape[0]
            hidden = h_0[:, :, :-1] # (1, bs, hidden_dim)
            t = h_0[:, :, -1].unsqueeze(-1) # (1, bs, 1)
            t_expanded = t + 1 + ptu.arange(0, L).view(L, 1, 1) # (L, bs, 1)
            cumsum = torch.cumsum(z, dim = 0) + hidden * t # (L, bs, hidden_size)
            t_emb = get_timestep_embedding(t_expanded.flatten(), embedding_dim=self.t_emb_size).reshape(L, -1, self.t_emb_size) # (L, bs, t_emb_size)
            output = torch.cat((cumsum / t_expanded, t_emb), dim = -1) # (L, bs, hidden_size + t_emb_size)
            h_n = torch.cat((output[-1, :, :self.hidden_size].unsqueeze(0), t+L), dim = -1) # (1, bs, hidden_size + 1)
        else:
            output = torch.cumsum(z, dim = 0) + h_0
            h_n = output[-1].unsqueeze(0)
        return output, h_n

    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((1, batch_size, self.hidden_size)).float() if self.agg == "sum" else ptu.zeros((1, batch_size, self.hidden_size+1)).float() # (h_t, t)





# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FFN layer Adopted from the PyTorch OpenAI GPT-2 model.
"""
class Block(nn.Module):
    def __init__(self, hidden_size, inner_size = None, dropout=0.0, layer_norm_epsilon=1e-5):
        super().__init__()
        inner_size = 4 * hidden_size if inner_size is None else inner_size
        self.ln = nn.LayerNorm(hidden_size, layer_norm_epsilon)
        self.mlp = MLP(hidden_size, inner_size, dropout=dropout)
    
    def forward(self, x):
        mlp_output = self.mlp(self.ln(x))
        x = x + mlp_output # residual connection
        return x



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.c_fc = Conv1D(hidden_dim, input_dim)
        self.c_proj = Conv1D(input_dim, hidden_dim)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)
    

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    


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