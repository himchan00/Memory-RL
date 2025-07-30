import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
import math
from torch import Tensor


class Hist(nn.Module):
    name = "hist"

    def __init__(self, input_size, hidden_size, n_layer, pdrop = 0.1, layer_norm_epsilon = 1e-5, **kwargs):
        """
        hyp_emb: If true, use hyperbolic embedding for the history representation
        """
        super().__init__()
        assert input_size == hidden_size, "input size must be equal to hidden size for residual connection"
        l_layer = [Block(hidden_size=hidden_size, dropout=pdrop, layer_norm_epsilon=layer_norm_epsilon) for _ in range(n_layer)]
        self.ln_in = nn.LayerNorm(input_size, layer_norm_epsilon)
        self.encoder = nn.Sequential(*l_layer)
        self.hidden_size = hidden_size
        self.num_layers = n_layer


    def forward(self, inputs, h_0):
        """
        inputs: (T, B, input_dim)
        h_0: (1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size)
        """
        z = self.encoder(self.ln_in(inputs))
        output = torch.cumsum(z, dim = 0) + h_0

        return output, output[-1].unsqueeze(0)

    def get_zero_internal_state(self, batch_size=1):
        return ptu.zeros((1, batch_size, self.hidden_size)).float()





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