import os
import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.optim import Adam, AdamW
from envs.alchemy import TRIAL_PHASE_DIM, valid_action_mask_from_observation
from policies.models.actor import CategoricalPolicy, TanhGaussianPolicy
from policies.models.aux_canon import AuxCanonMixin
from policies.models.off_policy_utils import (
    clip_gradients,
    prepare_recurrent_batch,
)
from policies.models.recurrent_head import RNN_head
from policies.models.popart import PopArt
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
from utils.helpers import get_constant_schedule_with_warmup


class ModelFreeOffPolicy_SAC_RNN(AuxCanonMixin, nn.Module):
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
        # SPIKE (throwaway): read the action type instead of hardcoding it.
        self.continuous_action = bool(kwargs.get("continuous_action", True))
        # SAC-discrete has no target actor (Christodoulou 2019).
        self.use_target_actor = self.continuous_action
        # Entropy target as a fraction of the maximum achievable entropy
        # (Christodoulou 2019 uses 0.98). Applied to log(action_dim), or to
        # log(legal actions) when invalid-action masking is on.
        self.discrete_target_entropy_ratio = float(
            getattr(config_rl, "discrete_target_entropy_ratio", 0.98)
        )
        self.compile_training_loss = bool(config_seq.get("compile", False))
        self._compiled_compute_loss = None
        self.mask_rl_loss_on_reset_transition = bool(
            config_seq.get("skip_reset_transition", False)
            and config_seq.get("mask_rl_loss_on_reset_transition", True)
        )

        ## Symbolic Alchemy invalid-action masking. Ported from
        # policy_rnn_dqn: most of the 40 actions name an absent stone or a
        # used-up potion at any given step, and every DQN number in the ledger
        # was measured with this on, so SAC-discrete needs it to be comparable.
        config_env = kwargs.get("config_env")
        is_alchemy = getattr(config_env, "env_type", None) == "alchemy"
        # Aux flags FIRST: net_obs_dim (obs minus the label block) is what the
        # network is built for, and the action-mask kwargs below slice against
        # it, not against the raw width.
        self.configure_aux_canon(config_rl, config_env, is_alchemy)
        self.mask_alchemy_invalid_actions = bool(
            getattr(config_rl, "mask_alchemy_invalid_actions", False)
            and is_alchemy
            and not self.continuous_action
        )
        self._alchemy_mask_kwargs = None
        if self.mask_alchemy_invalid_actions:
            observe_used = bool(getattr(config_env, "observe_used", True))
            add_trial_flag = bool(getattr(config_env, "add_trial_flag", False))
            structured_potions = bool(
                getattr(config_env, "structured_potions", False)
            )
            add_trial_phase = bool(getattr(config_env, "add_trial_phase", False))
            from envs.alchemy import get_symbolic_alchemy_layout

            layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
            symbolic_obs_dim = (
                layout.symbolic_obs_dim
                + int(add_trial_flag)
                + (TRIAL_PHASE_DIM if add_trial_phase else 0)
            )
            self._alchemy_mask_kwargs = {
                "observe_used": observe_used,
                "add_trial_flag": add_trial_flag,
                "context_dim": self.net_obs_dim - symbolic_obs_dim,
                "structured_potions": structured_potions,
                "add_trial_phase": add_trial_phase,
                "mask_no_op": bool(
                    getattr(config_rl, "mask_alchemy_no_op", False)
                ),
            }

        self.head = RNN_head(
            self.net_obs_dim,
            action_dim,
            config_seq,
        )
        self.build_aux_canon_head(config_rl)
        self.alternating_msc = bool(self.head.alternating_msc)
        # NOTE: no target head. Following amago

        self.qf1, self.qf2 = self.build_critic(
            input_size=self.head.embedding_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
            continuous_action=self.continuous_action,
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
            continuous_action=self.continuous_action,
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
            self._rl_parameters = tuple(self._get_parameters()) + (
                tuple(self.aux_canon_head.parameters())
                if self.aux_canon_head is not None else ()
            )
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
            if config_rl.target_entropy is not None:
                self.target_entropy = float(config_rl.target_entropy)
            elif self.continuous_action:
                self.target_entropy = -float(action_dim)
            else:
                # SAC-discrete: current_log_probs is sum_a pi log pi (negative
                # entropy), so the dual drives entropy -> target_entropy and the
                # target must be POSITIVE. 0.98 * log(A), Christodoulou 2019.
                # With invalid-action masking on, _compute_loss overrides this
                # per batch using the count of LEGAL actions instead of A.
                import math
                self.target_entropy = self.discrete_target_entropy_ratio * math.log(
                    float(action_dim)
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

    def sample_random_action(
        self,
        *,
        raw_obs: torch.Tensor | None = None,
        batch_shape=None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Uniform over LEGAL actions, returned one-hot.

        The Learner calls this for warm-up rollouts whenever the agent
        advertises `mask_alchemy_invalid_actions`, so a discrete SAC that turns
        masking on must provide it or the run dies before its first update.
        Same contract as the DQN entry point: (..., action_dim) float one-hot.
        """
        if raw_obs is not None:
            # The Learner hands us the RAW observation, labels included.
            raw_obs = self.strip_aux_target(raw_obs)
            batch_shape = raw_obs.shape[:-1]
            device = raw_obs.device
        if batch_shape is None:
            raise ValueError(
                "sample_random_action requires raw_obs or batch_shape"
            )
        if device is None:
            device = ptu.device

        mask = self._valid_action_mask(raw_obs)
        if mask is None:
            action = torch.randint(
                high=self.action_dim, size=tuple(batch_shape), device=device
            )
        else:
            # argmax of masked noise == uniform over the legal set, and needs
            # no per-row renormalisation.
            scores = torch.rand((*batch_shape, self.action_dim), device=device)
            action = torch.argmax(scores.masked_fill(~mask, -1.0), dim=-1)
        return F.one_hot(action.long(), num_classes=self.action_dim).float()

    def _valid_action_mask(self, raw_obs):
        """Boolean (..., A) mask of legal actions, or None when masking is off."""
        if not self.mask_alchemy_invalid_actions or raw_obs is None:
            return None
        mask = valid_action_mask_from_observation(
            raw_obs, **self._alchemy_mask_kwargs
        )
        if mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"Alchemy action mask width {mask.shape[-1]} does not match "
                f"action_dim {self.action_dim}"
            )
        return mask

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, continuous_action=True, **kwargs):
        policy_class = TanhGaussianPolicy if continuous_action else CategoricalPolicy
        return policy_class(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None,
                     continuous_action=True):
        assert action_dim is not None
        if obs_dim is not None:
            input_size = obs_dim
        # Continuous: Q(s, a) takes the action as input and emits one value.
        # Discrete: Q(s, .) emits one value per action, so the head can be
        # contracted against pi(.|s) without enumerating actions.
        critic_in = input_size + action_dim if continuous_action else input_size
        critic_out = 1 if continuous_action else action_dim
        qf1 = FlattenMlp(
            input_size=critic_in,
            output_size=critic_out,
            hidden_sizes=hidden_sizes,
        )
        qf2 = FlattenMlp(
            input_size=critic_in,
            output_size=critic_out,
            hidden_sizes=hidden_sizes,
        )
        return qf1, qf2

    def select_action(self, actor, observ, deterministic: bool, valid_mask=None):
        kwargs = {} if valid_mask is None else {"valid_mask": valid_mask}
        return actor(
            observ,
            deterministic=deterministic,
            return_log_prob=False,
            **kwargs,
        )[0]

    def forward_actor(self, actor, observ, valid_mask=None):
        if not self.continuous_action:
            # SAC-discrete: the "action" the loss contracts against is the full
            # probability vector, and log_prob is the full log-pi vector, so
            # sum_a pi(a|s)[Q(s,a) - alpha log pi(a|s)] needs no sampling.
            _, probs, log_probs = actor(
                observ, deterministic=False, return_log_prob=True,
                valid_mask=valid_mask,
            )
            return probs, log_probs
        action, mean, log_std, log_prob = actor(
            observ,
            reparameterize=True,
            deterministic=False,
            return_log_prob=True,
        )
        if log_prob is not None and log_prob.ndim == action.ndim:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def forward_actor_in_target(self, actor, actor_target, next_observ,
                                valid_mask=None):
        return self.forward_actor(actor_target, next_observ, valid_mask)

    def entropy_bonus(self, log_probs):
        return self.alpha_entropy * (-log_probs)

    @staticmethod
    def forward_frozen_critic(critic, observ):
        parameters = {
            name: parameter.detach()
            for name, parameter in critic.named_parameters()
        }
        return functional_call(critic, parameters, (observ,))

    def update_others(self, current_log_probs, target_entropy=None):
        if target_entropy is None:
            target_entropy = self.target_entropy
        if self.update_temperature:
            alpha_entropy_loss = -self.log_alpha_entropy.exp() * (
                current_log_probs + target_entropy
            )
            self.alpha_entropy_optim.zero_grad()
            alpha_entropy_loss.backward()
            self.alpha_entropy_optim.step()
            self.alpha_entropy = self.log_alpha_entropy.exp().detach()

        if os.environ.get("MATE_DEBUG_ALPHA"):
            # Temporary instrumentation: the collapse mode for masked
            # SAC-discrete is alpha going to zero, which crashes nothing and
            # only shows up as a policy that always picks NO_OP.
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 200 == 1:
                a = float(self.alpha_entropy)
                te = float(target_entropy)
                print(f"[alpha] step {self._dbg:6d}  alpha {a:.6f}  "
                      f"entropy {-float(current_log_probs):.4f}  target {te:.4f}",
                      flush=True)
        return {
            "entropy": -current_log_probs,
            "coef": self.alpha_entropy.squeeze(),
            "target_entropy": (
                target_entropy.detach()
                if torch.is_tensor(target_entropy)
                else torch.as_tensor(target_entropy)
            ),
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
        # Peel the labels off FIRST, then strip: nothing below this line sees
        # the answer key.
        aux_canon_targets = (
            self.aux_target_slice(observs) if self.aux_canon_enabled else None
        )
        observs = self.strip_aux_target(observs)
        loss_mask = masks
        if self.mask_rl_loss_on_reset_transition and memory_mask is not None:
            loss_mask = masks * memory_mask

        joint_embeds, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs, masks=masks,
            pos_offset=pos_offset, memory_mask=memory_mask,
        )
        target_joint_embeds = joint_embeds.detach()

        # (T+2, B, A), aligned 1:1 with joint_embeds; None when masking is off.
        valid_action_mask = self._valid_action_mask(observs)
        # With masking, log(action_dim) is NOT reachable: only a handful of the
        # 40 actions are legal at any step, so a target of 0.98*log(40)=3.62
        # exceeds the maximum entropy of the actual distribution and the dual
        # would drive alpha up without bound. Scale to the legal count instead.
        step_target_entropy = None
        if valid_action_mask is not None:
            n_valid = valid_action_mask.sum(dim=-1).clamp(min=1).to(
                joint_embeds.dtype
            )
            step_target_entropy = (
                self.discrete_target_entropy_ratio * torch.log(n_valid)
            ).mean()


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
                valid_mask=valid_action_mask,
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
        q1_pred_norm = self.popart(q1_pred_raw)
        q2_pred_norm = self.popart(q2_pred_raw)

        qf1_elementwise = F.huber_loss(
            q1_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf2_elementwise = F.huber_loss(
            q2_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf1_elementwise = qf1_elementwise * loss_mask
        qf2_elementwise = qf2_elementwise * loss_mask
        num_valid_per_timestep = loss_mask.sum(dim=(1, 2)).clamp(min=1.0)
        qf1_loss = qf1_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        qf2_loss = qf2_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep

        ### 3. Actor loss
        new_actions, new_log_probs = self.forward_actor(
            actor=self.policy, observ=joint_embeds,
            valid_mask=valid_action_mask,
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

        policy_elementwise = policy_loss[:-1] * loss_mask
        policy_loss = (
            policy_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        )

        ### 4. update
        qf_loss = 0.5 * (qf1_loss + qf2_loss)
        num_valid = loss_mask.sum().clamp(min=1.0)
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
            "q1": ((q1_pred_denorm * loss_mask).sum() / num_valid).detach(),
            "q2": ((q2_pred_denorm * loss_mask).sum() / num_valid).detach(),
            "actor_loss": actor_loss.detach(),
            "policy_loss": policy_loss.detach(),
        }
        # Seq-model aux loss (e.g. MSC; training-only); non-detached, so pop before logging.
        aux_loss = d_forward.pop("_aux_loss", None)
        # Pop the exposed tensors BEFORE outputs.update(d_forward). The Learner
        # stacks every metric across all updates in a rollout, so leaving a
        # (T+2, B, 256) memory readout in there costs ~17 GB per rollout and
        # OOMs. Underscore keys are the caller's responsibility to remove.
        memory_embeds = d_forward.pop("_memory_embeds", None)
        encoded_obs = d_forward.pop("_encoded_obs", None)
        if self.alternating_msc and aux_loss is not None:
            raise RuntimeError(
                "Alternating MSC RL forward unexpectedly returned _aux_loss; "
                "MSC loss must be optimized through update_msc()"
            )
        outputs.update(d_forward)

        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            outputs["aux_loss"] = aux_loss.detach()

        if self.aux_canon_enabled:
            aux_canon_loss, aux_canon_metrics = self.aux_canon_loss(
                self.aux_canon_embeds(joint_embeds, memory_embeds, encoded_obs),
                observs, aux_canon_targets, loss_mask,
            )
            total_loss = total_loss + self.aux_canon_weight * aux_canon_loss
            outputs.update(aux_canon_metrics)

        if step_target_entropy is not None:
            # Leading underscore: popped in forward() before logging.
            outputs["_target_entropy"] = step_target_entropy.detach()

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
            loss_mask = masks
            if self.mask_rl_loss_on_reset_transition and memory_mask is not None:
                loss_mask = masks * memory_mask
            current_log_probs = (new_log_probs[:-1] * loss_mask).sum() / num_valid
            current_log_probs = current_log_probs.detach()
        outputs.update(
            self.update_others(
                current_log_probs,
                target_entropy=outputs.pop("_target_entropy", None),
            )
        )
        
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
        # LEAK GUARD: strip before RNN_head, the critic or the action mask run.
        prev_obs = self.strip_aux_target(prev_obs).unsqueeze(0)
        obs = self.strip_aux_target(obs).unsqueeze(0)

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
        # obs is (1, B, dim) here; the mask must match joint_embed's (B, ·).
        valid_mask = self._valid_action_mask(obs[-1])
        current_action = self.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
            valid_mask=valid_mask,
        )

        return current_action, current_internal_state
