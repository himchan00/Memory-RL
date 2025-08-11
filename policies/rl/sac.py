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

    # ---------- losses ----------
    def critic_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        dones,
        gamma,
        next_observs=None,  # used in markov_critic
    ):
        """
        y = r + (1-d)*γ * [ min(Q1',Q2')(s', a'~π) - α * log π(a'|s') ]
        Shapes follow your SACD conventions:
          - Markov: (B, ·)
          - Non-Markov (shared RNN): (T+1,B,·) and we align to (T,B,·)
        """
        with torch.no_grad():
            # a' ~ π(·|s')
            if markov_actor:
                next_a, next_logp = self.forward_actor(
                    actor, next_observs if markov_critic else observs
                )
            else:
                next_a, next_logp = actor(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=next_observs if markov_critic else observs,
                    reparameterize=True,
                    deterministic=False,
                    return_log_prob=True,
                )
                if next_logp.ndim == next_a.ndim:
                    next_logp = next_logp.sum(dim=-1, keepdim=True)

            # Q targets at (s', a')
            if markov_critic:
                qa = torch.cat([next_observs, next_a], dim=-1)     # (B, obs+act)
                q1_t = critic_target[0](qa)
                q2_t = critic_target[1](qa)
            else:
                # sequence critic returns (T+1,B,1)
                q1_t, q2_t = critic_target(
                    prev_actions=actions,
                    rewards=rewards,
                    observs=observs,
                    current_actions=next_a,
                )

            min_next_q = torch.min(q1_t, q2_t) - self.alpha_entropy * next_logp
            q_target = rewards + (1.0 - dones) * gamma * min_next_q
            if not markov_critic:
                q_target = q_target[1:]  # align to (T,B,1)

        # current Q(s,a)
        if markov_critic:
            qa_now = torch.cat([observs, actions], dim=-1)
            q1 = critic[0](qa_now)
            q2 = critic[1](qa_now)
        else:
            q1, q2 = critic(
                prev_actions=actions[:-1],
                rewards=rewards[:-1],
                observs=observs[:-1],
                current_actions=actions[1:],   # stored a_t
            )

        return (q1, q2), q_target

    def actor_loss(
        self,
        markov_actor: bool,
        markov_critic: bool,
        actor,
        actor_target,
        critic,
        critic_target,
        observs,
        actions=None,
        rewards=None,
    ):
        # sample a ~ π(·|s)
        if markov_actor:
            a_pi, logp_pi = self.forward_actor(actor, observs)
        else:
            a_pi, logp_pi = actor(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                reparameterize=True,
                deterministic=False,
                return_log_prob=True,
            )
            if logp_pi.ndim == a_pi.ndim:
                logp_pi = logp_pi.sum(dim=-1, keepdim=True)

        # Q(s,a)
        if markov_critic:
            qa = torch.cat([observs, a_pi], dim=-1)
            q1 = critic[0](qa)
            q2 = critic[1](qa)
        else:
            q1, q2 = critic(
                prev_actions=actions,
                rewards=rewards,
                observs=observs,
                current_actions=a_pi,
            )

        min_q = torch.min(q1, q2)
        policy_loss = self.alpha_entropy * logp_pi - min_q
        if not markov_critic:
            policy_loss = policy_loss[:-1]  # align to (T,B,1)

        # for temperature update
        current_log_probs = logp_pi
        return policy_loss, current_log_probs

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
