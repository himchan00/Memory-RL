import torch
import torchkit.pytorch_utils as ptu
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
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
    
    def norm(self, x, scale=True):
        y = torch.clamp((x - self.mean) / torch.sqrt(self.var + self.epsilon), -5.0, 5.0) if scale else (x - self.mean)
        return y



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