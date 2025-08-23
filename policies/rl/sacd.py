import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from .base import RLAlgorithmBase
from policies.models.actor import CategoricalPolicy
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu


class SACD(RLAlgorithmBase):
    name = "sacd"
    continuous_action = False
    use_target_actor = False

    def __init__(
        self,
        init_temperature=0.1,
        update_temperature=True,
        target_entropy=None,
        temp_lr=3e-4,
        action_dim=None,
        **kwargs,
    ):

        self.update_temperature = update_temperature
        if self.update_temperature:
            assert target_entropy is not None
            self.target_entropy = float(target_entropy) * np.log(action_dim)
            self.log_alpha_entropy = torch.zeros(
                1, requires_grad=True, device=ptu.device
            )
            self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=temp_lr)
            self.alpha_entropy = self.log_alpha_entropy.exp().detach().item()
        else:
            self.alpha_entropy = init_temperature

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return CategoricalPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        qf1 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        qf2 = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool):
        return actor(observ, deterministic, return_log_prob=False)[0]

    @staticmethod
    def forward_actor(actor, observ):
        _, probs, log_probs = actor(observ, return_log_prob=True)
        return probs, log_probs  # (T+1, B, dim), (T+1, B, dim)


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

    #### Below are used in shared RNN setting
    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)
