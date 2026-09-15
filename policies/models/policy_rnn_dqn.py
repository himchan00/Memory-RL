import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from policies.models.off_policy_utils import (
    clip_gradients,
    prepare_recurrent_batch,
)
from policies.models.alchemy_aux import AlchemyAux
from policies.models.recurrent_head import RNN_head
from policies.models.popart import PopArt
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
from utils.helpers import get_constant_schedule_with_warmup


class LinearSchedule:
    def __init__(self, init_value: float, end_value: float, transition_steps: int):
        self.init = float(init_value)
        self.end = float(end_value)
        self.n = max(1, int(transition_steps))

    def __call__(self, step: int) -> float:
        t = 0 if step < 0 else self.n if step > self.n else step
        frac = t / self.n
        return (1.0 - frac) * self.init + frac * self.end


class ModelFreeOffPolicy_DQN_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        config_env=None,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm
        self.compile_training_loss = bool(config_seq.get("compile", False))
        self._compiled_compute_loss = None

        critic_loss = str(getattr(config_rl, "critic_loss", "mse")).lower()
        if critic_loss not in ("mse", "huber"):
            raise ValueError(
                f'config_rl.critic_loss must be "mse" or "huber", got {critic_loss!r}'
            )
        self.critic_loss_name = critic_loss
        self._critic_loss_fn = (
            F.mse_loss if critic_loss == "mse" else F.huber_loss
        )

        self.epsilon_schedule = LinearSchedule(
            init_value=config_rl.init_eps,
            end_value=config_rl.end_eps,
            transition_steps=config_rl.schedule_steps,
        )
        self.count = 0

        # Symbolic Alchemy: the env appends a supervision label to the
        # observation, so the network must be sized for the STRIPPED width and
        # the label must be excised before anything can read it. A first pass
        # settles that width; the heads need `self.head` and are built after.
        self.alchemy = AlchemyAux(
            obs_dim, action_dim, config_rl, config_env, config_seq
        ) if config_env is not None else None
        net_obs_dim = (
            self.alchemy.net_obs_dim if self.alchemy is not None else obs_dim
        )
        self.net_obs_dim = net_obs_dim

        # Shared RNN encoder
        self.head = RNN_head(net_obs_dim, action_dim, config_seq, config_env)
        if self.alchemy is not None:
            self.alchemy.build_heads(self.head)
        self.alternating_msc = bool(self.head.alternating_msc)
        # NOTE: no target head. Following amago

        # Q-value network
        self.qf = FlattenMlp(
            input_size=self.head.embedding_size,
            output_size=action_dim,
            hidden_sizes=config_rl.config_critic.hidden_dims,
        )
        self.qf_target = deepcopy(self.qf)

        # PopArt value normalization (no-op when disabled)
        self.popart = PopArt(
            beta=getattr(config_rl, "popart_beta", 5e-4),
            init_nu=getattr(config_rl, "popart_init_nu", 100.0),
            enabled=getattr(config_rl, "use_popart", False),
        )

        # Optimizer
        if self.alternating_msc:
            self._rl_parameters = (
                *self.head.rl_parameters(),
                *self.qf.parameters(),
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
            self._rl_parameters = (
                *self.head.parameters(),
                *self.qf.parameters(),
            )
            self._msc_parameters = ()

        self.critic_optimizer = AdamW(
            self._rl_parameters,
            lr=config_rl.critic_lr,
        )
        # reference to https://github.com/UT-Austin-RPL/amago/blob/main/amago/experiment.py
        self.lr_schedule = get_constant_schedule_with_warmup(
            optimizer=self.critic_optimizer, num_warmup_steps=500
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
        prev_obs = prev_obs.unsqueeze(0)        # (1, B, dim)
        obs = obs.unsqueeze(0)                  # (1, B, dim)

        # Excise the supervision label BEFORE anything can read it.
        if self.alchemy is not None:
            prev_obs = self.alchemy.strip_target(prev_obs)
            obs = self.alchemy.strip_target(obs)

        joint_embed, current_internal_state, transition_embedding = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial,
            timestep=timestep,
        )

        valid_mask = (
            self.alchemy.valid_action_mask(obs.squeeze(0))
            if self.alchemy is not None else None
        )
        current_action = self._select_action(joint_embed, deterministic, valid_mask)

        return current_action, current_internal_state, transition_embedding

    def _select_action(self, observ, deterministic: bool, valid_mask=None):
        batch_size = observ.shape[0]
        action_logits = self.qf(observ)
        if valid_mask is not None:
            # Both the greedy pick and the epsilon draw must stay inside the
            # legal set; an illegal action is a wasted step, not exploration.
            action_logits = AlchemyAux.mask_logits(action_logits, valid_mask)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            if valid_mask is not None:
                scores = torch.rand(
                    action_logits.shape, device=action_logits.device
                ).log()
                random_action = AlchemyAux.mask_logits(
                    scores, valid_mask
                ).argmax(dim=-1)
            else:
                random_action = torch.randint(
                    high=action_logits.shape[-1],
                    size=action_logits.shape[:-1],
                ).to(ptu.device)
            optimal_action = torch.argmax(action_logits, dim=-1)

            eps = self.epsilon_schedule(self.count)
            mask = torch.multinomial(
                input=ptu.FloatTensor([1 - eps, eps]),
                num_samples=action_logits.shape[0],
                replacement=True,
            )
            action = mask * random_action + (1 - mask) * optimal_action
            self.count += batch_size

        return F.one_hot(
            action.long(),
            num_classes=action_logits.shape[-1],
        ).float()

    @torch.no_grad()
    def encode_transition_embedding(self, action, reward, observ, next_observ):
        """STORE's rollout cache, with the label stripped first.

        The Learner used to reach past the agent into `head` for this, which is
        the one path in the Alchemy port that never met `strip_target`. It only
        surfaced as a crash (259 vs 193 features) because the label widens the
        transition; had the widths matched, the privileged 33-dim answer key
        would have been cached straight into the memory.
        """
        if self.alchemy is not None:
            observ = self.alchemy.strip_target(observ)
            next_observ = self.alchemy.strip_target(next_observ)
        return self.head.encode_transition_embedding(
            action, reward, observ, next_observ
        )

    @property
    def mask_alchemy_invalid_actions(self):
        """Read by Learner._sample_random_action to route the warm-up draw."""
        return self.alchemy is not None and self.alchemy.mask_invalid

    def sample_random_action(self, *, raw_obs=None):
        action = self.alchemy.random_action(self.alchemy.strip_target(raw_obs))
        if action is None:
            raise RuntimeError("sample_random_action called without a mask")
        return action

    def _compute_loss(
        self, actions, rewards, observs, next_observs, terms, masks,
        transition_t, cached_embeddings=None, cached_prefixes=None, *,
        reuse_shared_observations=False,
    ):
        """
        For physical replay row j_t = transition_t[t]:
        actions[t]      = a_{j_t-1}, shape (L, B, A) one-hot
        rewards[t]      = r_{j_t-1}, shape (L, B, 1)
        observs[t]      = s_{j_t-1}, shape (L, B, obs_dim)
        next_observs[t] = s_{j_t},   shape (L, B, obs_dim)
        terms[t]        = done_{j_t-1}, shape (L, B, 1)
        masks[t]        = mask_{j_t-1}, shape (L, B, 1)
        """
        aux_targets = None
        if self.alchemy is not None and self.alchemy.target_enabled:
            # The label rides on the observation; take it, then cut it out so
            # RNN_head, the critic and the action mask only ever see the
            # stripped width the network was built for.
            aux_targets = self.alchemy.target_slice(observs)
            observs = self.alchemy.strip_target(observs)
            next_observs = self.alchemy.strip_target(next_observs)

        ### 1. Compute embeddings once
        current_joint, next_joint, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs,
            next_observs=next_observs, masks=masks, transition_t=transition_t,
            reuse_shared_observations=reuse_shared_observations,
            cached_embeddings=cached_embeddings,
            cached_prefixes=cached_prefixes,
        )  # each (L, B, dim)
        ### 2. Critic loss (DDQN)
        # Current Q values (raw / pre-POP-affine)
        q_pred_all_raw = self.qf(current_joint)  # (L, B, A)

        with torch.no_grad():
            # DDQN: online net selects next action, target net evaluates its value
            next_q_online_raw = self.qf(next_joint)
            # The selection MUST respect the action mask. Alchemy leaves most of
            # its 40 actions illegal at any step, and an illegal action's Q is
            # never updated by experience, so bootstrapping through one lets the
            # target grow without bound -- measured q climbing 41 -> 1988 while
            # the task's own ceiling is 315.
            next_valid_mask = (
                self.alchemy.valid_action_mask(next_observs)
                if self.alchemy is not None else None
            )
            next_q_online_raw = AlchemyAux.mask_logits(
                next_q_online_raw, next_valid_mask
            )
            next_actions = torch.argmax(next_q_online_raw, dim=-1, keepdim=True)  # (L, B, 1)
            next_q_target_raw = self.qf_target(next_joint.detach())  # (L, B, A)
            next_q_raw = next_q_target_raw.gather(-1, next_actions)  # (L, B, 1)
            next_q_denorm = self.popart(next_q_raw, normalized=False)  # denorm → reward scale
            q_target_denorm = rewards + (1.0 - terms) * self.gamma * next_q_denorm  # (L, B, 1) reward scale
            self.popart.update_stats(q_target_denorm, masks)
            q_target_norm = self.popart.normalize_values(q_target_denorm)

        # Gather Q(s_{j_t-1}, M_t, a_{j_t-1}) from current pair embeddings.
        actions_idx = torch.argmax(actions, dim=-1, keepdim=True)  # (L, B, 1)
        q_pred_raw = q_pred_all_raw.gather(-1, actions_idx)  # (L, B, 1)

        # Apply POP affine (w*x + b) before Bellman residual so stats shifts preserve gradient signal.
        q_pred_norm = self.popart(q_pred_raw)
        qf_elementwise = self._critic_loss_fn(
            q_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf_elementwise = qf_elementwise * masks
        num_valid_per_timestep = masks.sum(dim=(1, 2)).clamp(min=1.0)
        qf_loss = qf_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        num_valid = masks.sum().clamp(min=1.0)
        critic_loss = qf_elementwise.sum() / num_valid

        # Denormalize for interpretable logging (critic outputs are raw / pre-affine)
        q_pred_denorm = self.popart(q_pred_raw, normalized=False)
        outputs = {
            "critic_loss": critic_loss.detach(),
            "qf_loss": qf_loss.detach(),
            "q": ((q_pred_denorm * masks).sum() / num_valid).detach(),
            "target_q": ((q_target_denorm * masks).sum() / num_valid).detach(),
        }
        # Seq-model aux loss (e.g. MSC; training-only); non-detached, so pop before logging.
        aux_loss = d_forward.pop("_aux_loss", None)
        if self.alternating_msc and aux_loss is not None:
            raise RuntimeError(
                "Alternating MSC RL forward unexpectedly returned _aux_loss; "
                "MSC loss must be optimized through update_msc()"
            )
        outputs.update(d_forward)

        ### 3. Update
        total_loss = critic_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            outputs["aux_loss"] = aux_loss.detach()

        ### 3b. Symbolic Alchemy auxiliary supervision (canonical frame).
        # Not an input: `aux_targets` was excised above. What it buys is the
        # INCENTIVE to compute the perceived -> latent frame map, which the
        # scalar TD signal alone never produces.
        if self.alchemy is not None and self.alchemy.enabled:
            memory_embeds = outputs.pop("_memory_embeds", None)
            encoded_obs = outputs.pop("_encoded_obs", None)
            aux_embeds = self.alchemy.site_embeds(
                current_joint, memory_embeds, encoded_obs
            )
            alchemy_loss, alchemy_metrics = self.alchemy.loss(
                aux_embeds, observs, aux_targets, masks
            )
            total_loss = total_loss + alchemy_loss
            outputs.update(alchemy_metrics)
        else:
            outputs.pop("_memory_embeds", None)
            outputs.pop("_encoded_obs", None)
        outputs.pop("_obs_embeds", None)

        return total_loss, outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf, self.qf_target, self.tau)

    def training_state_dict(self):
        state_dict = {
            "model": self.state_dict(),
            "optimizer": self.critic_optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict(),
            "count": self.count,
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
        self.critic_optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_schedule.load_state_dict(state_dict["lr_schedule"])
        if self.alternating_msc:
            self.aux_optimizer.load_state_dict(state_dict["aux_optimizer"])
            self.aux_lr_schedule.load_state_dict(
                state_dict["aux_lr_schedule"]
            )
        self.count = int(state_dict["count"])

    def update(self, batch):
        is_subset = batch.get("sample_mode") == "subset"
        recurrent_batch = prepare_recurrent_batch(
            batch,
            discrete_action_dim=self.action_dim,
        )

        compute_loss = self._compute_loss
        if self.compile_training_loss and recurrent_batch.actions.is_cuda:
            if self._compiled_compute_loss is None:
                self._compiled_compute_loss = torch.compile(
                    self._compute_loss,
                    dynamic=False,
                )
            compute_loss = self._compiled_compute_loss

        total_loss, outputs = compute_loss(
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

        self.critic_optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            outputs.update(
                clip_gradients(
                    self._rl_parameters,
                    self.clip_grad_norm,
                )
            )

        self.critic_optimizer.step()
        self.lr_schedule.step()

        ### 4. Soft update
        self.soft_target_update()

        return outputs

    def update_msc(self, batch):
        if not self.alternating_msc:
            raise RuntimeError(
                "update_msc requires alternating_ema MSC mode"
            )

        recurrent_batch = prepare_recurrent_batch(
            batch,
            discrete_action_dim=self.action_dim,
        )
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
