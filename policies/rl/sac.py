# policies/rl/sac.py
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

from .base import RLAlgorithmBase
from policies.models.actor import TanhGaussianPolicy   
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu


class SAC(RLAlgorithmBase):
    name = "sac"
    continuous_action = True
    use_target_actor = False

    def __init__(
        self,
        init_temperature=0.1,
        update_temperature=True,
        target_entropy=None,   # if None -> -action_dim
        temp_lr=3e-4,
        action_dim=None,
        **kwargs,
    ):
        self.update_temperature = update_temperature

        if self.update_temperature:
            assert action_dim is not None, "action_dim required for auto temperature"
            self.target_entropy = -float(action_dim) if target_entropy is None \
                                  else float(target_entropy)
            self.log_alpha_entropy = torch.zeros(
                1, requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=temp_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = init_temperature

    # ---------- builders ----------
    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        # returns tanh-Gaussian policy with SAC-friendly API
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        # two scalar Q(s,a) networks (Markov case)
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        qf1 = FlattenMlp(
            input_size=input_size + action_dim, output_size=1, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size + action_dim, output_size=1, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    # ---------- actor helpers ----------
    def select_action(self, actor, observ, deterministic: bool):
        # (B, act_dim) or (T+1,B,act_dim) depending on observ shape
        return actor(observ, deterministic=deterministic, return_log_prob=False)[0]

    @staticmethod
    def forward_actor(actor, observ):
        """
        Return (actions, log_probs) to parallel SACD’s (probs, log_probs).
        log_probs is summed over action dims -> shape (..., 1)
        """
        action, mean, log_std, log_prob = actor(
            observ, reparameterize=True, deterministic=False, return_log_prob=True
        )
        if log_prob is not None and log_prob.ndim == action.ndim:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


    # ---------- temperature ----------
    def update_others(self, current_log_probs):
        if self.update_temperature:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().item()

        return {"entropy": -current_log_probs, "coef": self.alpha_entropy}

    # ---------- helpers kept for symmetry with SACD ----------
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)
