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
        self.continuous_action = True
        self.use_target_actor = True
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
        )
        # reference to https://github.com/UT-Austin-RPL/amago/blob/main/amago/experiment.py
        self.lr_schedule = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=50000 
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
            )
            self.aux_lr_schedule = get_constant_schedule_with_warmup(
                optimizer=self.aux_optimizer,
                num_warmup_steps=50000,
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

    def forward_actor_in_target(self, actor, actor_target, next_observ):
        return self.forward_actor(actor_target, next_observ)

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
        self, actions, rewards, observs, terms, masks, pos_offset=None,
        memory_mask=None,
    ):
        """
        actions[t] = a_{t-1}, shape (T+1, B, dim)
        rewards[t] = r_{t-1}, shape (T+1, B, dim)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        terms[t] = done_{t-1}, shape (T+1, B, 1)
        masks[t] = mask_{t-1}, shape (T+1, B, 1)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == terms.dim()
            == observs.dim()
            == masks.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0]
            == terms.shape[0]
            == observs.shape[0] - 1
            == masks.shape[0]
        )
        length, batch_size, _ = actions.shape
        loss_mask = masks if memory_mask is None else masks * memory_mask

        joint_embeds, joint_embeds_target, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs, masks=masks,
            pos_offset=pos_offset, memory_mask=memory_mask,
        )
        if joint_embeds_target is None:
            joint_embeds_target = joint_embeds
        target_joint_embeds = joint_embeds_target.detach()


        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            new_next_actions, new_next_log_probs = self.forward_actor_in_target(
                actor=self.policy,
                actor_target=(
                    self.policy_target
                    if self.use_target_actor
                    else self.policy
                ),
                next_observ=(
                    target_joint_embeds
                    if self.use_target_actor
                    else joint_embeds
                ),
            )

            if self.continuous_action:
                target_joint_embeds = torch.cat(
                    (target_joint_embeds, new_next_actions),
                    dim=-1,
                )
            # super_sac convention: add entropy_bonus in raw (pre-affine) space, then denormalize.
            next_q1_raw = self.qf1_target(target_joint_embeds)  # (T+1,B,1) if cont_act else (T+1,B,A)
            next_q2_raw = self.qf2_target(target_joint_embeds)
            min_next_q_target_raw = torch.min(next_q1_raw, next_q2_raw)
            entropy_bonus = self.entropy_bonus(new_next_log_probs)
            min_next_q_target_raw = min_next_q_target_raw + entropy_bonus
            if not self.continuous_action:
                min_next_q_target_raw = (
                    new_next_actions * min_next_q_target_raw
                ).sum(dim=-1, keepdims=True)
            min_next_q_target_raw = min_next_q_target_raw[1:]  # (T+1,B,1)
            min_next_q_target_denorm = self.popart(min_next_q_target_raw, normalized=False)
            q_target_denorm = rewards + (1.0 - terms) * self.gamma * min_next_q_target_denorm
            self.popart.update_stats(q_target_denorm, loss_mask)
            q_target_norm = self.popart.normalize_values(q_target_denorm)

        # Q(h(t), a(t)) (T, B, 1)
        if self.continuous_action:
            curr_joint_embeds = torch.cat(
                (joint_embeds[:-1], actions),
                dim=-1,
            )
        else:
            curr_joint_embeds = joint_embeds[:-1]

        q1_pred_raw = self.qf1(curr_joint_embeds)
        q2_pred_raw = self.qf2(curr_joint_embeds)
        if not self.continuous_action:
            actions_idx = torch.argmax(actions, dim=-1, keepdims=True)
            q1_pred_raw = q1_pred_raw.gather(dim=-1, index=actions_idx)
            q2_pred_raw = q2_pred_raw.gather(dim=-1, index=actions_idx)

        # Apply POP affine (w*x + b) before Bellman residual so stats shifts preserve gradient signal.
        q1_pred_norm = self.popart(q1_pred_raw) * loss_mask
        q2_pred_norm = self.popart(q2_pred_raw) * loss_mask
        q_target_norm = q_target_norm * loss_mask

        # PopArt normalizes targets to ~unit variance, so MSE (amago default) is appropriate.
        # Fall back to Huber when PopArt is off to retain outlier robustness.
        if self.popart.enabled:
            qf1_loss = (q1_pred_norm - q_target_norm).pow(2).mean(dim=(1, 2))
            qf2_loss = (q2_pred_norm - q_target_norm).pow(2).mean(dim=(1, 2))
        else:
            qf1_loss = torch.nn.HuberLoss(reduction='none')(
                q1_pred_norm, q_target_norm
            ).mean(dim=(1, 2))
            qf2_loss = torch.nn.HuberLoss(reduction='none')(
                q2_pred_norm, q_target_norm
            ).mean(dim=(1, 2))

        ### 3. Actor loss
        new_actions, new_log_probs = self.forward_actor(
            actor=self.policy, observ=joint_embeds
        )

        if self.freeze_critic:
            joint_embeds = joint_embeds.detach()
        if self.continuous_action:
            new_joint_embeds = torch.cat(
                (joint_embeds, new_actions),
                dim=-1,
            )
        else:
            new_joint_embeds = joint_embeds

        # Actor sees normalized Q (w*x + b); entropy bonus is scaled by w to match the target's σ·w·α weight in reward space.
        if self.freeze_critic:
            q1_pi_raw = self.forward_frozen_critic(
                self.qf1,
                new_joint_embeds,
            )
            q2_pi_raw = self.forward_frozen_critic(
                self.qf2,
                new_joint_embeds,
            )
        else:
            q1_pi_raw = self.qf1(new_joint_embeds)
            q2_pi_raw = self.qf2(new_joint_embeds)
        q1_pi_norm = self.popart(q1_pi_raw)
        q2_pi_norm = self.popart(q2_pi_raw)

        min_q_new_actions_norm = torch.min(q1_pi_norm, q2_pi_norm)  # (T+1,B,1) or (T+1,B,A)
        policy_loss = -min_q_new_actions_norm
        entropy_loss = -self.entropy_bonus(new_log_probs) * self.popart.w
        policy_loss += entropy_loss

        if not self.continuous_action:
            policy_loss = (new_actions * policy_loss).sum(
                axis=-1, keepdims=True
            )
            new_log_probs = (new_actions * new_log_probs).sum(
                axis=-1, keepdims=True
            )

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = policy_loss * loss_mask
        policy_loss = policy_loss.mean(dim=(1, 2))  # (T,)

        ### 4. update
        qf_loss = 0.5 * (qf1_loss + qf2_loss)
        total_loss = (qf_loss + policy_loss).mean()

        num_valid = torch.clamp(loss_mask.sum(), min=1.0) # for logging exact average q values
        # Denormalize predicted Q for interpretable logging (critic outputs are raw / pre-affine)
        q1_pred_denorm = self.popart(q1_pred_raw, normalized=False)
        q2_pred_denorm = self.popart(q2_pred_raw, normalized=False)
        outputs = {
            "critic_loss": qf_loss.mean().detach(),
            "qf_loss": qf_loss.detach(),
            "q1": ((q1_pred_denorm * loss_mask).sum() / num_valid).detach(),
            "q2": ((q2_pred_denorm * loss_mask).sum() / num_valid).detach(),
            "actor_loss": policy_loss.mean().detach(),
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

        return total_loss, new_log_probs, num_valid, outputs

    def forward(
        self, actions, rewards, observs, terms, masks, pos_offset=None,
        memory_mask=None,
    ):
        compute_loss = self._compute_loss
        if self.compile_training_loss and actions.is_cuda:
            if self._compiled_compute_loss is None:
                self._compiled_compute_loss = torch.compile(
                    self._compute_loss,
                    dynamic=False,
                )
            compute_loss = self._compiled_compute_loss

        total_loss, new_log_probs, num_valid, outputs = compute_loss(
            actions,
            rewards,
            observs,
            terms,
            masks,
            pos_offset,
            memory_mask,
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
        with torch.no_grad():
            loss_mask = masks if memory_mask is None else masks * memory_mask
            current_log_probs = (new_log_probs[:-1] * loss_mask).sum() / num_valid
            current_log_probs = current_log_probs.detach()
        outputs.update(self.update_others(current_log_probs))
        
        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.use_target_actor:
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
        recurrent_batch = prepare_recurrent_batch(batch)
        actions = recurrent_batch.actions
        if not self.continuous_action:
            actions = F.one_hot(
                actions.squeeze(-1).long(),
                num_classes=self.action_dim,
            ).float()

        return self.forward(
            actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.terms,
            recurrent_batch.masks,
            recurrent_batch.pos_offset,
            recurrent_batch.memory_mask,
        )

    def update_msc(self, batch):
        if not self.alternating_msc:
            raise RuntimeError(
                "update_msc requires alternating_ema MSC mode"
            )

        recurrent_batch = prepare_recurrent_batch(batch)
        actions = recurrent_batch.actions
        if not self.continuous_action:
            actions = F.one_hot(
                actions.squeeze(-1).long(),
                num_classes=self.action_dim,
            ).float()

        raw_loss, outputs = self.head.compute_msc_loss(
            actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.masks,
            recurrent_batch.memory_mask,
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
        skip_memory_update=False,
    ):

        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        prev_reward = prev_reward.unsqueeze(0)  # (1, B, 1)
        prev_obs = prev_obs.unsqueeze(0)  # (1, B, dim)
        obs = obs.unsqueeze(0) # (1, B, dim)

        joint_embed, current_internal_state = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial,
            timestep=timestep,
            skip_memory_update=skip_memory_update,
        )

        # 4. Actor head, generate action tuple
        current_action = self.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
