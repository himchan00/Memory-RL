"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}


class Mlp(nn.Module):
    """
    Multi-layer perceptron network
    dropout is applied to input as in gpt
    fc layer - norm - activation - dropout is common and effective ordering and used here
    norm: one of ["none", "layer", "batch"]
    """
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation="leakyrelu",
        output_activation="linear",
        norm = "none",
        dropout=0,
        project_output = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)
        self.norm = norm
        assert self.norm in ["none", "layer", "batch"]
        self.fcs = []
        self.norms = []
        self.dropout = nn.Dropout(dropout)
        self.project_output = project_output
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)
            if self.norm == "layer":
                ln = nn.LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.norms.append(ln)
            elif self.norm == "batch":
                bn = nn.BatchNorm1d(next_size)
                self.__setattr__("batch_norm{}".format(i), bn)
                self.norms.append(bn)
            else:
                self.norms.append(nn.Identity())

        self.last_fc = nn.Linear(in_size, output_size)
        if self.norm == "layer":
            ln = nn.LayerNorm(output_size)
            self.__setattr__("layer_norm_final", ln)
            self.norms.append(ln)
        elif self.norm == "batch":
            bn = nn.BatchNorm1d(output_size)
            self.__setattr__("batch_norm_final", bn)
            self.norms.append(bn)
        else:
            self.norms.append(nn.Identity())


    def forward(self, input):
        input_shape = input.shape
        input = input.view(-1, input_shape[-1])  # flatten except for last dim
        h = self.dropout(input)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.norms[i](h)
            h = self.hidden_activation(h)
            h = self.dropout(h)
        preactivation = self.last_fc(h)
        preactivation = self.norms[-1](preactivation)
        output = self.output_activation(preactivation)
        output = self.dropout(output)
        output = output.view(*input_shape[:-1], self.output_size)  # restore shape
        if self.project_output:
            output = output / output.norm(dim=-1, keepdim=True).clamp(min=1e-6) * np.sqrt(self.output_size)
        return output


class double_Mlp(nn.Module):
    """
    Apply Mlp1 to first input_size of input, and Mlp2 to the rest, then concatenate outputs.
    """
    
    def __init__(self, mlp1, mlp2):
        super().__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.input_size = mlp1.input_size + mlp2.input_size
        self.output_size = mlp1.output_size + mlp2.output_size

    def forward(self, input):
        assert input.shape[-1] == self.input_size, f"Input size mismatch: expected {self.input_size}, got {input.shape[-1]}"
        input_1 = input[..., :self.mlp1.input_size]
        input_2 = input[..., self.mlp1.input_size:]
        output_1 = self.mlp1(input_1)  # (T, B, mlp1.output_size)
        output_2 = self.mlp2(input_2)  # (T, B, mlp2.output_size)
        output = torch.cat((output_1, output_2), dim=-1)  # (T, B, output_size)
        return output



from transformers.modeling_utils import Conv1D
from transformers.activations import ACT2FN

class gpt_like_Mlp(nn.Module):
    """
    gpt2-like MLP network for ablation study
    """
    
    def __init__(self, hidden_size, n_layer, pdrop, use_output_ln = True):
        super().__init__()
        self.input_drop = nn.Dropout(pdrop)
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                gpt_mlp_block(hidden_size, activation="gelu_new", pdrop=pdrop)
                for _ in range(n_layer)
            ]
        )
        if use_output_ln:
            self.output_ln = nn.LayerNorm(hidden_size) # gpt2 uses layer norm at the output
        else:
            self.output_ln = nn.Identity() # mate uses rms norm, so we may skip layer norm here

    def forward(self, x):
        """
        x: (T, B, input_size)
        return: (T, B, input_size)
        """
        h = self.input_drop(x)
        for layer in self.layers:
            h = layer(h)
        out = self.output_ln(h)
        return out


class gpt_mlp_block(nn.Module):
    """
    A residual MLP block used in gpt2 architecture. Refer to MLP class in policies/seq_models/trajectory_gpt2.py. (Not imported to avoid namespace conflict)
    """

    def __init__(self, hidden_size, activation="gelu_new", pdrop=0.1):
        super().__init__()
        self.c_fc = Conv1D(4 * hidden_size, hidden_size)
        self.c_proj = Conv1D(hidden_size, 4 * hidden_size)
        self.act = ACT2FN[activation]
        self.dropout = nn.Dropout(pdrop)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x_residual = self.ln(x)
        x_residual = self.dropout(self.c_proj(self.act(self.c_fc(x_residual))))
        x = x + x_residual
        return x 
    

class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along last dim
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    from math import floor

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
    )
    w = floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    return h, w


class ImageEncoder(nn.Module):
    def __init__(
        self,
        image_shape,
        embedding_size=100,
        channels=[8, 16],
        kernel_sizes=[2, 2],
        strides=[1, 1],
        activation=relu_name,
        from_flattened=True,
        normalize_pixel=True,
        **kwargs,
    ):
        super(ImageEncoder, self).__init__()
        self.shape = image_shape
        self.channels = [image_shape[0]] + list(channels)

        layers = []
        h_w = self.shape[-2:]

        for i in range(len(self.channels) - 1):
            layers.append(
                nn.Conv2d(
                    self.channels[i], self.channels[i + 1], kernel_sizes[i], strides[i]
                )
            )
            layers.append(ACTIVATIONS[activation]())
            h_w = conv_output_shape(h_w, kernel_sizes[i], strides[i])

        self.cnn = nn.Sequential(*layers)

        self.linear = nn.Linear(
            h_w[0] * h_w[1] * self.channels[-1], embedding_size
        )  # dreamer does not use it

        self.from_flattened = from_flattened
        self.normalize_pixel = normalize_pixel
        self.embedding_size = embedding_size

    def forward(self, image):
        # return embedding of shape [N, embedding_size]
        if self.from_flattened:
            # image of size (T, B, C*H*W)
            batch_size = image.shape[:-1]
            img_shape = [np.prod(batch_size)] + list(self.shape)  # (T*B, C, H, W)
            image = torch.reshape(image, img_shape)
        else:  # image of size (N, C, H, W)
            batch_size = [image.shape[0]]

        if self.normalize_pixel:
            image = image / 255.0

        embed = self.cnn(image)  # (T*B, C, H, W)

        embed = torch.reshape(embed, list(batch_size) + [-1])  # (T, B, C*H*W)
        embed = self.linear(embed)  # (T, B, embedding_size)
        return embed



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
    elif s_act == 'gelu':
        return ACT2FN['gelu_new']
    else:
        raise ValueError(f'Unexpected activation: {s_act}')