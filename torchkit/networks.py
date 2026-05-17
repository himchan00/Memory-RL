"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F

relu_name = "relu"
elu_name = "elu"
ACTIVATIONS = {
    relu_name: nn.ReLU,
    elu_name: nn.ELU,
}


class IdentityModule(nn.Module):
    """nn.Identity that ignores extra kwargs (e.g. ``mask``) for drop-in use as
    an embedder placeholder."""

    def forward(self, x, **kwargs):
        return x


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
        normalize_inputs=False,
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
        self.fcs = nn.ModuleList()
        self.in_norm = InputNorm(input_size, skip=not normalize_inputs)
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.project_output = project_output
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.fcs.append(fc)
            if self.norm == "layer":
                ln = nn.LayerNorm(next_size)
                self.norms.append(ln)
            elif self.norm == "batch":
                bn = nn.BatchNorm1d(next_size)
                self.norms.append(bn)
            else:
                self.norms.append(nn.Identity())

        self.last_fc = nn.Linear(in_size, output_size)
        if self.norm == "layer":
            ln = nn.LayerNorm(output_size)
            self.norms.append(ln)
        elif self.norm == "batch":
            bn = nn.BatchNorm1d(output_size)
            self.norms.append(bn)
        else:
            self.norms.append(nn.Identity())


    def forward(self, input, mask=None):
        if self.training:
            self.in_norm.update_stats(input, mask=mask)
        input = self.in_norm(input)
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
        if self.project_output:
            output = F.normalize(output, p=2, dim=-1) * np.sqrt(self.output_size)
        return output


class double_Mlp(nn.Module):
    """
    Apply Mlp1 to first input_size of input, and Mlp2 to the rest, then concatenate outputs.
    """
    
    def __init__(self, mlp1, mlp2, input_size1, input_size2):
        super().__init__()
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.input_size1 = input_size1
        self.input_size2 = input_size2

    def forward(self, input, mask=None):
        assert input.shape[-1] == self.input_size1 + self.input_size2, f"Input size mismatch: expected {self.input_size1 + self.input_size2}, got {input.shape[-1]}"
        input_1 = input[..., :self.input_size1]
        input_2 = input[..., self.input_size1:]
        output_1 = self.mlp1(input_1, mask=mask)  # (T, B, mlp1.output_size)
        output_2 = self.mlp2(input_2, mask=mask)  # (T, B, mlp2.output_size)
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
    



class InputNorm(nn.Module):
    """Moving-average feature normalization, imported from amago/nets/utils.py. 

    Normalizes input features using a moving average of their statistics. This
    helps stabilize training by keeping the input distribution relatively
    constant.

    Args:
        dim: Dimension of the input feature.

    Keyword Args:
        beta: Smoothing parameter for the moving average. Defaults to 1e-4.
        init_nu: Initial value for the moving average of the squared feature
            values. Defaults to 1.0.
        skip (no gin): Whether to skip normalization. Defaults to False. Cannot be
            configured via gin (disable input norm in the TstepEncoder config).
    """

    def __init__(self, dim, beta=1e-4, init_nu=1.0, skip: bool = False):
        super().__init__()
        self.skip = skip
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("nu", torch.ones(dim) * init_nu)
        self.register_buffer("_t", torch.ones((1,)))
        self.beta = beta
        self.pad_val = 4.0

    @property
    def sigma(self):
        sigma_ = torch.sqrt(self.nu - self.mu**2 + 1e-5)
        return torch.nan_to_num(sigma_).clamp(1e-3, 1e6)

    def normalize_values(self, val: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return val
        sigma = self.sigma
        normalized = ((val - self.mu) / sigma).clamp(-1e4, 1e4)
        not_nan = ~torch.isnan(normalized)
        stable = (sigma > 0.01).expand_as(not_nan)
        use_norm = torch.logical_and(stable, not_nan)
        output = torch.where(use_norm, normalized, (val - torch.nan_to_num(self.mu)))
        return output

    def denormalize_values(self, val: torch.Tensor) -> torch.Tensor:
        if self.skip:
            return val
        sigma = self.sigma
        denormalized = (val * sigma) + self.mu
        stable = (sigma > 0.01).expand_as(denormalized)
        output = torch.where(stable, denormalized, (val + torch.nan_to_num(self.mu)))
        return output

    def masked_stats(self, val: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # If an explicit mask is given, use it. Otherwise fall back to pad_val match.
        # mask shape is expected to be broadcastable to val[..., :1] (e.g. (T, B, 1)).
        if mask is None:
            mask = (~((val == self.pad_val).all(-1, keepdim=True))).float()
        else:
            mask = mask.to(val.dtype)
            if mask.dim() == val.dim() - 1:
                mask = mask.unsqueeze(-1)
        sum_ = (val * mask).sum((0, 1))
        square_sum = ((val * mask) ** 2).sum((0, 1))
        total = mask.sum((0, 1))
        mean = sum_ / total
        square_mean = square_sum / total
        return mean, square_mean

    def update_stats(self, val: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        self._t += 1
        old_sigma = self.sigma
        old_mu = self.mu
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        mean, square_mean = self.masked_stats(val, mask=mask)
        self.mu.data = (1.0 - beta_t) * self.mu + (beta_t * mean)
        self.nu.data = (1.0 - beta_t) * self.nu + (beta_t * square_mean)

    def forward(self, x: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
        if denormalize:
            return self.denormalize_values(x)
        else:
            return self.normalize_values(x)