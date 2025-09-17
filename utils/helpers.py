import numpy as np
import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torch.optim.lr_scheduler import LambdaLR
from functools import partial


def get_grad_norm(model):
    # mean of grad norm^2
    grad_norm = []
    for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        grad_norm.append(p.grad.data.norm(2).item())
    if grad_norm:
        grad_norm = np.mean(grad_norm)
    else:
        grad_norm = 0.0
    return grad_norm



class FeatureExtractor(nn.Module):
    """one-layer MLP with relu
    Used for extracting features for vector-based observations/actions/rewards

    NOTE: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch.linear is a linear transformation in the LAST dimension
    with weight of size (IN, OUT)
    which means it can support the input size larger than 2-dim, in the form
    of (*, IN), and then transform into (*, OUT) with same size (*)
    e.g. In the encoder, the input is (N, B, IN) where N=seq_len.
    """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return ptu.zeros(
                0,
            )  # useful for concat


import torchkit.pytorch_utils as ptu

class RunningMeanStd(object):
    # Adopted from https://github.com/jacooba/hyper/blob/main/utils/helpers.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    # Modified with reference to https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/running_mean_std.py
    def __init__(self, epsilon=1e-5, shape=(), init_count=0):
        self.mean = ptu.zeros(shape).float()
        self.var = ptu.ones(shape).float()
        self.epsilon = epsilon
        self.count = ptu.ones(()).float() * init_count

    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
    
    def norm(self, x):
        y = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return torch.clamp(y, -5.0, 5.0) # clip to avoid numerical issues

    def denorm(self, x):
        x = x.clamp(-5.0, 5.0)
        return x * torch.sqrt(self.var + self.epsilon) + self.mean


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def _get_constant_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1
):
    """Get a constant learning rate schedule with a warmup period."""
    lr_lambda = partial(
        _get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)