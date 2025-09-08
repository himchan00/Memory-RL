"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn

from torchkit.core import PyTorchModule
from torchkit.modules import LayerNorm

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        hidden_activation="leakyrelu",
        output_activation="linear",
        norm = "none",
        norm_mode = "final", # Where to apply normalization and dropout
        dropout=0,
        identity=False, # Indicator for Identity network
    ):
        self.save_init_params(locals())
        super().__init__()
        self.identity = identity    
        if self.identity:
            self.input_size = input_size
            self.output_size = input_size
            self.hidden_sizes = None
        else:
            self.input_size = input_size
            self.output_size = output_size
            self.hidden_sizes = hidden_sizes
            self.hidden_activation = get_activation(hidden_activation)
            self.output_activation = get_activation(output_activation)
            self.norm = norm
            assert self.norm in ["none", "layer", "spectral"]
            assert norm_mode in ["all", "final", "all_but_final"]
            self.norm_mode = norm_mode
            self.fcs = []
            self.layer_norms = []
            self.dropout = nn.Dropout(dropout)
            in_size = input_size

            for i, next_size in enumerate(hidden_sizes):
                fc = nn.Linear(in_size, next_size)
                if self.norm == "spectral" and (self.norm_mode in ["all", "all_but_final"]):
                    fc = nn.utils.spectral_norm(fc)
                in_size = next_size
                self.__setattr__("fc{}".format(i), fc)
                self.fcs.append(fc)

                if self.norm == "layer" and (self.norm_mode in ["all", "all_but_final"]):
                    ln = LayerNorm(next_size)
                    self.__setattr__("layer_norm{}".format(i), ln)
                    self.layer_norms.append(ln)

            self.last_fc = nn.Linear(in_size, output_size)
            if self.norm_mode in ["all", "final"]:
                if self.norm == "spectral":
                    self.last_fc = nn.utils.spectral_norm(self.last_fc)
                if self.norm == "layer":
                    ln = LayerNorm(output_size)
                    self.__setattr__("layer_norm_final", ln)
                    self.layer_norms.append(ln)

    def forward(self, input):
        if self.identity:
            output = input
        else:
            h = input
            for i, fc in enumerate(self.fcs):
                h = fc(h)
                if self.norm == "layer" and (self.norm_mode in ["all", "all_but_final"]):
                    h = self.layer_norms[i](h)
                h = self.hidden_activation(h)
                h = self.dropout(h)
            preactivation = self.last_fc(h)
            if self.norm_mode in ["all", "final"] and self.norm == "layer":
                preactivation = self.layer_norms[-1](preactivation)
            output = self.output_activation(preactivation)
            if self.norm_mode in ["all", "final"]:
                output = self.dropout(output)
        return output


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
    else:
        raise ValueError(f'Unexpected activation: {s_act}')