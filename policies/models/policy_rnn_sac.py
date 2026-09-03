import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.optim import Adam, AdamW
from policies.models.actor import TanhGaussianPolicy
from policies.models.off_policy_utils import (
    clip_gradients,
    prepare_recurrent_batch,
)
from policies.models.recurrent_head import RNN_head
from policies.models.popart import PopArt
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
from utils.helpers import get_constant_schedule_with_warmup


class ModelFreeOffPolicy_SAC_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    We find `freeze_critic = True` can prevent degradation shown in https://github.com/twni2016/pomdp-baselines
    """

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        freeze_critic: bool,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm
        self.freeze_critic = freeze_critic
        self.compile_training_loss = bool(config_seq.get("compile", False))
        self._compiled_compute_loss = None

        self.head = RNN_head(
            obs_dim,
            action_dim,
            config_seq,
        )
        self.alternating_msc = bool(self.head.alternating_msc)
        # NOTE: no target head. Following amago

        self.qf1, self.qf2 = self.build_critic(
            input_size=self.head.embedding_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # PopArt value normalization (no-op when disabled)
        self.popart = PopArt(
            beta=getattr(config_rl, "popart_beta", 5e-4),
            init_nu=getattr(config_rl, "popart_init_nu", 100.0),
            enabled=getattr(config_rl, "use_popart", False),
        )

        # policy network
        self.policy = self.build_actor(
            input_size=self.head.embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=config_rl.config_actor.hidden_dims,
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        assert config_rl.critic_lr == config_rl.actor_lr
        if self.alternating_msc:
            self._rl_parameters = (
                *self.head.rl_parameters(),
                *self.qf1.parameters(),
                *self.qf2.parameters(),
                *self.policy.parameters(),
            )
            self._msc_parameters = tuple(self.head.msc_parameters())
            if not self._rl_parameters:
                raise ValueError("Alternating MSC requires RL parameters")
            if not self._msc_parameters:
                raise ValueError("Alternating MSC requires MSC parameters")
            if not {
                id(param) for param in self._rl_parameters
            }.isdisjoint(id(param) for param in self._msc_parameters):
                raise ValueError(
                    "Alternating MSC RL and MSC parameter lists must be disjoint"
                )
        else:
            self._rl_parameters = tuple(self._get_parameters())
            self._msc_parameters = ()

        self.optimizer = AdamW(
            self._rl_parameters,
            lr=config_rl.critic_lr,
            weight_decay=0.001,
        )
        # reference to https://github.com/UT-Austin-RPL/amago/blob/main/amago/experiment.py
        self.lr_schedule = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=500
        )
        if self.alternating_msc:
            msc_lr = float(
                config_seq.seq_model.get("msc_lr", config_rl.critic_lr)
            )
            if msc_lr <= 0.0:
                raise ValueError(
                    "config_seq.seq_model.msc_lr must be positive"
                )
            self.aux_optimizer = AdamW(
                self._msc_parameters,
                lr=msc_lr,
                weight_decay=0.001,
            )
            self.aux_lr_schedule = get_constant_schedule_with_warmup(
                optimizer=self.aux_optimizer,
                num_warmup_steps=500,
            )

        self.update_temperature = config_rl.update_temperature
        if self.update_temperature:
            self.target_entropy = (
                -float(action_dim)
                if config_rl.target_entropy is None
                else float(config_rl.target_entropy)
            )
            self.log_alpha_entropy = torch.zeros(
                1,
                requires_grad=True,
                device=ptu.device,
            )
            self.alpha_entropy_optim = Adam(
                [self.log_alpha_entropy],
                lr=config_rl.temp_lr,
            )
            self.alpha_entropy = self.log_alpha_entropy.exp().detach()
        else:
            self.alpha_entropy = config_rl.get("init_temperature", 0.1)

    def _get_parameters(self):
        # exclude targets
        params = [
            *self.head.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
            *self.policy.parameters(),
        ]
        return params

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
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
            input_size=input_size + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
        )
        qf2 = FlattenMlp(
            input_size=input_size + action_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool):
        return actor(
            observ,
            deterministic=deterministic,
            return_log_prob=False,
        )[0]

    @staticmethod
    def forward_actor(actor, observ):
        action, mean, log_std, log_prob = actor(
            observ,
            reparameterize=True,
            deterministic=False,
            return_log_prob=True,
        )
        if log_prob is not None and log_prob.ndim == action.ndim:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def forward_actor_in_target(self, next_observ):
        return self.forward_actor(self.policy_target, next_observ)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)

    @staticmethod
    def forward_frozen_critic(critic, observ):
        parameters = {
            name: parameter.detach()
            for name, parameter in critic.named_parameters()
        }
        return functional_call(critic, parameters, (observ,))

    def update_others(self, current_log_probs):
        if self.update_temperature:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + self.target_entropy
            )
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().detach()

        return {
            "entropy": -current_log_probs,
            "coef": self.alpha_entropy.squeeze(),
        }


    def _compute_loss(
        self, actions, rewards, observs, next_observs, terms, masks,
        transition_t, cached_embeddings=None, cached_prefixes=None, *,
        reuse_shared_observations=False,
    ):
        """
        For physical replay row j_t = transition_t[t]:
        actions[t]      = a_{j_t-1}, shape (L, B, action_dim)
        rewards[t]      = r_{j_t-1}, shape (L, B, 1)
        observs[t]      = s_{j_t-1}, shape (L, B, obs_dim)
        next_observs[t] = s_{j_t},   shape (L, B, obs_dim)
        terms[t]        = done_{j_t-1}, shape (L, B, 1)
        masks[t]        = mask_{j_t-1}, shape (L, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == observs.dim()
            == next_observs.dim()
            == terms.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[:2]
            == rewards.shape[:2]
            == observs.shape[:2]
            == next_observs.shape[:2]
            == terms.shape[:2]
            == masks.shape[:2]
        )

        current_joint, next_joint, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs,
            next_observs=next_observs, masks=masks, transition_t=transition_t,
            reuse_shared_observations=reuse_shared_observations,
            cached_embeddings=cached_embeddings,
            cached_prefixes=cached_prefixes,
        )  # each (L, B, dim)

        ### 2. Critic loss

        # Q^tar(s_{j_t}, M_{t+1}, pi(s_{j_t}, M_{t+1})) + H[pi]
        with torch.no_grad():
            target_next_joint = next_joint.detach()
            new_next_actions, new_next_log_probs = self.forward_actor_in_target(
                next_observ=target_next_joint,
            )
            target_critic_inputs = torch.cat(
                (target_next_joint, new_next_actions),
                dim=-1,
            )
            next_q1_raw = self.qf1_target(target_critic_inputs)  # (L, B, 1)
            next_q2_raw = self.qf2_target(target_critic_inputs)
            min_next_q_target_raw = torch.min(next_q1_raw, next_q2_raw)
            # super_sac convention: add entropy_bonus in raw (pre-affine) space, then denormalize.
            min_next_q_target_raw = min_next_q_target_raw + self.entropy_bonus(new_next_log_probs)
            min_next_q_target_denorm = self.popart(min_next_q_target_raw, normalized=False)
            q_target_denorm = rewards + (1.0 - terms) * self.gamma * min_next_q_target_denorm
            self.popart.update_stats(q_target_denorm, masks)
            q_target_norm = self.popart.normalize_values(q_target_denorm)

        # Q(s_{j_t-1}, M_t, a_{j_t-1}) (L, B, 1)
        critic_inputs = torch.cat(
            (current_joint, actions),
            dim=-1,
        )
        q1_pred_raw = self.qf1(critic_inputs)
        q2_pred_raw = self.qf2(critic_inputs)

        # Apply POP affine (w*x + b) before Bellman residual so stats shifts preserve gradient signal.
        q1_pred_norm = self.popart(q1_pred_raw)
        q2_pred_norm = self.popart(q2_pred_raw)

        qf1_elementwise = F.mse_loss(
            q1_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf2_elementwise = F.mse_loss(
            q2_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf1_elementwise = qf1_elementwise * masks
        qf2_elementwise = qf2_elementwise * masks
        num_valid_per_timestep = masks.sum(dim=(1, 2)).clamp(min=1.0)
        qf1_loss = qf1_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        qf2_loss = qf2_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep

        ### 3. Actor loss
        new_actions, new_log_probs = self.forward_actor(
            actor=self.policy, observ=current_joint
        )

        actor_joint = current_joint.detach() if self.freeze_critic else current_joint
        actor_inputs = torch.cat(
            (actor_joint, new_actions),
            dim=-1,
        )

        # Actor sees normalized Q (w*x + b); entropy bonus is scaled by w to match the target's σ·w·α weight in reward space.
        if self.freeze_critic:
            q1_pi_raw = self.forward_frozen_critic(
                self.qf1,
                actor_inputs,
            )
            q2_pi_raw = self.forward_frozen_critic(
                self.qf2,
                actor_inputs,
            )
        else:
            q1_pi_raw = self.qf1(actor_inputs)
            q2_pi_raw = self.qf2(actor_inputs)
        q1_pi_norm = self.popart(q1_pi_raw)
        q2_pi_norm = self.popart(q2_pi_raw)

        policy_elementwise = (
            -torch.min(q1_pi_norm, q2_pi_norm)
            - self.entropy_bonus(new_log_probs) * self.popart.w
        )  # (L, B, 1)
        policy_elementwise = policy_elementwise * masks
        policy_loss = (
            policy_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        )

        ### 4. update
        qf_loss = 0.5 * (qf1_loss + qf2_loss)
        num_valid = masks.sum().clamp(min=1.0)
        critic_loss = 0.5 * (
            qf1_elementwise.sum() + qf2_elementwise.sum()
        ) / num_valid
        actor_loss = policy_elementwise.sum() / num_valid
        total_loss = critic_loss + actor_loss

        # Denormalize predicted Q for interpretable logging (critic outputs are raw / pre-affine)
        q1_pred_denorm = self.popart(q1_pred_raw, normalized=False)
        q2_pred_denorm = self.popart(q2_pred_raw, normalized=False)
        outputs = {
            "critic_loss": critic_loss.detach(),
            "qf_loss": qf_loss.detach(),
            "q1": ((q1_pred_denorm * masks).sum() / num_valid).detach(),
            "q2": ((q2_pred_denorm * masks).sum() / num_valid).detach(),
            "actor_loss": actor_loss.detach(),
            "policy_loss": policy_loss.detach(),
        }
        # Seq-model aux loss (e.g. MSC; training-only); non-detached, so pop before logging.
        aux_loss = d_forward.pop("_aux_loss", None)
        if self.alternating_msc and aux_loss is not None:
            raise RuntimeError(
                "Alternating MSC RL forward unexpectedly returned _aux_loss; "
                "MSC loss must be optimized through update_msc()"
            )
        outputs.update(d_forward)

        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            outputs["aux_loss"] = aux_loss.detach()

        mean_log_prob = (new_log_probs.detach() * masks).sum() / num_valid
        return total_loss, mean_log_prob, outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def training_state_dict(self):
        temperature_state = {
            "alpha_entropy": (
                self.alpha_entropy.detach().cpu()
                if torch.is_tensor(self.alpha_entropy)
                else self.alpha_entropy
            ),
        }
        if self.update_temperature:
            temperature_state.update(
                {
                    "log_alpha_entropy": self.log_alpha_entropy.detach().cpu(),
                    "optimizer": self.alpha_entropy_optim.state_dict(),
                }
            )
        state_dict = {
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict(),
            "temperature": temperature_state,
        }
        if self.alternating_msc:
            state_dict.update(
                {
                    "aux_optimizer": self.aux_optimizer.state_dict(),
                    "aux_lr_schedule": self.aux_lr_schedule.state_dict(),
                }
            )
        return state_dict

    def load_training_state_dict(self, state_dict):
        self.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_schedule.load_state_dict(state_dict["lr_schedule"])
        if self.alternating_msc:
            self.aux_optimizer.load_state_dict(state_dict["aux_optimizer"])
            self.aux_lr_schedule.load_state_dict(
                state_dict["aux_lr_schedule"]
            )

        temperature_state = state_dict["temperature"]
        alpha_entropy = temperature_state["alpha_entropy"]
        self.alpha_entropy = (
            alpha_entropy.to(ptu.device)
            if torch.is_tensor(alpha_entropy)
            else alpha_entropy
        )
        if self.update_temperature:
            self.log_alpha_entropy.data.copy_(
                temperature_state["log_alpha_entropy"].to(ptu.device)
            )
            self.alpha_entropy_optim.load_state_dict(
                temperature_state["optimizer"]
            )

    def update(self, batch):
        is_subset = batch.get("sample_mode") == "subset"
        recurrent_batch = prepare_recurrent_batch(batch)
        compute_loss = self._compute_loss
        if self.compile_training_loss and recurrent_batch.actions.is_cuda:
            if self._compiled_compute_loss is None:
                self._compiled_compute_loss = torch.compile(
                    self._compute_loss,
                    dynamic=False,
                )
            compute_loss = self._compiled_compute_loss

        total_loss, mean_log_prob, outputs = compute_loss(
            recurrent_batch.actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.next_observs,
            recurrent_batch.terms,
            recurrent_batch.masks,
            recurrent_batch.transition_t,
            recurrent_batch.cached_embeddings,
            recurrent_batch.cached_prefixes,
            reuse_shared_observations=not is_subset,
        )

        outputs.update(self.popart.metrics())

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            outputs.update(
                clip_gradients(
                    self._rl_parameters,
                    self.clip_grad_norm,
                )
            )

        self.optimizer.step()
        self.lr_schedule.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        outputs.update(self.update_others(mean_log_prob))

        return outputs

    def update_msc(self, batch):
        if not self.alternating_msc:
            raise RuntimeError(
                "update_msc requires alternating_ema MSC mode"
            )

        recurrent_batch = prepare_recurrent_batch(batch)

        raw_loss, outputs = self.head.compute_msc_loss(
            recurrent_batch.actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.next_observs,
            recurrent_batch.masks,
        )

        self.aux_optimizer.zero_grad()
        raw_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_metrics = clip_gradients(
                self._msc_parameters,
                self.clip_grad_norm,
            )
            outputs.update(
                {
                    f"msc_{key}": value
                    for key, value in grad_metrics.items()
                }
            )

        self.aux_optimizer.step()
        self.aux_lr_schedule.step()
        self.head.update_msc_ema(self.tau)

        return outputs

    
    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        prev_reward,
        prev_obs,
        obs,
        deterministic=False,
        initial=False,
        timestep=0,
    ):

        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        prev_reward = prev_reward.unsqueeze(0)  # (1, B, 1)
        prev_obs = prev_obs.unsqueeze(0)  # (1, B, dim)
        obs = obs.unsqueeze(0) # (1, B, dim)

        joint_embed, current_internal_state, transition_embedding = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial,
            timestep=timestep,
        )

        # 4. Actor head, generate action tuple
        current_action = self.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state, transition_embedding
