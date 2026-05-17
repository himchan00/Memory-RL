import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from policies.rl import RL_ALGORITHMS
from policies.models.recurrent_head import RNN_head
from policies.models.popart import PopArt
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

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        self.head = RNN_head(
            obs_dim,
            action_dim,
            config_seq,
        )
        # NOTE: no target head. Following amago

        # q-value networks
        # NOTE: For continuous SAC, algo.build_critic will internally expect [input_size + action_dim].
        # For discrete SACD, it returns (A)-dim outputs from just [input_size].
        self.qf1, self.qf2 = self.algo.build_critic(
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
        self.policy = self.algo.build_actor(
            input_size=self.head.embedding_size,
            action_dim=self.action_dim,
            hidden_sizes=config_rl.config_actor.hidden_dims,
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        assert config_rl.critic_lr == config_rl.actor_lr
        self.optimizer = AdamW(self._get_parameters(), lr=config_rl.critic_lr)
        # reference to https://github.com/UT-Austin-RPL/amago/blob/main/amago/experiment.py
        self.lr_schedule = get_constant_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=50000 
        )

    def _get_parameters(self):
        # exclude targets
        params = [
            *self.head.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
            *self.policy.parameters(),
        ]
        return params


    def forward(self, actions, rewards, observs, terms, masks):
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

        joint_embeds, d_forward = self.head.forward(actions=actions, rewards=rewards, observs=observs, masks=masks)
        target_joint_embeds = joint_embeds.detach()

        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim for continuous) or (T+1, B, A for discrete probs)
            # new_next_log_probs: (T+1, B, 1) for continuous OR (T+1, B, A) for discrete
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(
                actor=self.policy,
                actor_target=self.policy_target if self.algo.use_target_actor else self.policy,
                next_observ=target_joint_embeds if self.algo.use_target_actor else joint_embeds
            )

            if self.algo.continuous_action:
                target_joint_embeds = torch.cat((target_joint_embeds, new_next_actions), dim = -1)
            # Compute target Q in denormalized (reward) space, as in vanilla SAC.
            # PopArt's effect: scale entropy bonus by sigma so its reward-scale weight (sigma * alpha)
            next_q1_denorm = self.popart(self.qf1_target(target_joint_embeds), normalized=False) # (T+1,B,1) if cont_act else (T+1,B,A)
            next_q2_denorm = self.popart(self.qf2_target(target_joint_embeds), normalized=False)
            min_next_q_target_denorm = torch.min(next_q1_denorm, next_q2_denorm) # (T+1,B,1) if cont_act else (T+1,B,A)
            entropy_bonus = self.algo.entropy_bonus(new_next_log_probs)
            if self.popart.enabled:
                entropy_bonus = entropy_bonus * self.popart.sigma
            min_next_q_target_denorm = min_next_q_target_denorm + entropy_bonus
            if not self.algo.continuous_action:
                min_next_q_target_denorm = (new_next_actions * min_next_q_target_denorm).sum(dim=-1, keepdims=True)
            min_next_q_target_denorm = min_next_q_target_denorm[1:]  # (T+1,B,1)
            q_target_denorm = rewards + (1.0 - terms) * self.gamma * min_next_q_target_denorm
            self.popart.update_stats(q_target_denorm, masks)
            q_target_norm = self.popart.normalize_values(q_target_denorm)

        # Q(h(t), a(t)) (T, B, 1)
        # 3. joint embeds
        if self.algo.continuous_action: # Continuous: Q(h_t, a_t) using stored actions (T,B,act_dim)
            curr_joint_embeds = torch.cat((joint_embeds[:-1], actions), dim = -1)
        else:
            curr_joint_embeds = joint_embeds[:-1]

        q1_pred_raw = self.qf1(curr_joint_embeds)
        q2_pred_raw = self.qf2(curr_joint_embeds)
        if not self.algo.continuous_action: # Discrete (original): gather on action id from logits (T,B,A)->(T,B,1)
            actions_idx = torch.argmax(actions, dim=-1, keepdims=True)  # (T,B,1)
            q1_pred_raw = q1_pred_raw.gather(dim=-1, index=actions_idx)  # (T,B,1)
            q2_pred_raw = q2_pred_raw.gather(dim=-1, index=actions_idx)  # (T,B,1)

        # Apply POP affine (w*x + b) before Bellman residual so stats shifts preserve gradient signal.
        q1_pred_norm = self.popart(q1_pred_raw) * masks
        q2_pred_norm = self.popart(q2_pred_raw) * masks
        q_target_norm = q_target_norm * masks

        # PopArt normalizes targets to ~unit variance, so MSE (amago default) is appropriate.
        # Fall back to Huber when PopArt is off to retain outlier robustness.
        if self.popart.enabled:
            qf1_loss = (q1_pred_norm - q_target_norm).pow(2).mean(dim=(1, 2))
            qf2_loss = (q2_pred_norm - q_target_norm).pow(2).mean(dim=(1, 2))
        else:
            qf1_loss = torch.nn.HuberLoss(reduction='none')(q1_pred_norm, q_target_norm).mean(dim=(1, 2))
            qf2_loss = torch.nn.HuberLoss(reduction='none')(q2_pred_norm, q_target_norm).mean(dim=(1, 2))

        ### 3. Actor loss
        # Continuous: J_pi = E[ alpha*logpi - minQ ]
        # Discrete:   E_{a~pi}[ Q + H ]
        new_actions, new_log_probs = self.algo.forward_actor(
            actor=self.policy, observ=joint_embeds
        )

        if self.freeze_critic:
            self._freeze_critic()
            joint_embeds = joint_embeds.detach()
        if self.algo.continuous_action:
            new_joint_embeds = torch.cat((joint_embeds, new_actions), dim = -1) # (T+1, B, dim)
        else:
            new_joint_embeds = joint_embeds

        # Following super_sac (online_actor_update): actor sees normalized Q (w*x + b) + natural-scale entropy.
        q1_pi_norm = self.popart(self.qf1(new_joint_embeds))
        q2_pi_norm = self.popart(self.qf2(new_joint_embeds))
        if self.freeze_critic:
            self._unfreeze_critic()

        min_q_new_actions_norm = torch.min(q1_pi_norm, q2_pi_norm)  # (T+1,B,1) or (T+1,B,A)
        policy_loss = -min_q_new_actions_norm
        entropy_loss = -self.algo.entropy_bonus(new_log_probs)
        policy_loss += entropy_loss

        if not self.algo.continuous_action:
            policy_loss = (new_actions * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)
            new_log_probs = (new_actions * new_log_probs).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = policy_loss * masks
        policy_loss = policy_loss.mean(dim=(1, 2))  # (T,)

        ### 4. update
        qf_loss = 0.5 * (qf1_loss + qf2_loss)
        total_loss = (qf_loss + policy_loss).mean()

        num_valid = torch.clamp(masks.sum(), min=1.0) # for logging exact average q values
        # Denormalize predicted Q for interpretable logging (critic outputs are raw / pre-affine)
        q1_pred_denorm = self.popart(q1_pred_raw, normalized=False)
        q2_pred_denorm = self.popart(q2_pred_raw, normalized=False)
        outputs = {
            "critic_loss": qf_loss.mean().detach(),
            "qf_loss": qf_loss.detach(),
            "q1": ((q1_pred_denorm * masks).sum() / num_valid).detach(),
            "q2": ((q2_pred_denorm * masks).sum() / num_valid).detach(),
            "actor_loss": policy_loss.mean().detach(),
            "policy_loss": policy_loss.detach(),
            "popart_mu": self.popart.mu.detach().mean(),
            "popart_sigma": self.popart.sigma.detach().mean(),
            "popart_w": self.popart.w.detach().mean(),
            "popart_b": self.popart.b.detach().mean(),

        }
        outputs.update(d_forward)

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm # Only clip gradients of the RNN head.
            )
            outputs["raw_grad_norm"] = grad_norm.detach()
            outputs["grad_clip_coef"] = torch.clamp(
                self.clip_grad_norm / (grad_norm.detach() + 1e-12), max=1.0
            )
            outputs["clip_grad_norm"] = self.clip_grad_norm

        self.optimizer.step()
        self.lr_schedule.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        if new_log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.detach()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)
        
        return outputs

    
    def _freeze_critic(self):
        for param in self.qf1.parameters():
            param.requires_grad = False
        for param in self.qf2.parameters():
            param.requires_grad = False

    def _unfreeze_critic(self):
        for param in self.qf1.parameters():
            param.requires_grad = True
        for param in self.qf2.parameters():
            param.requires_grad = True

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)


    def update(self, batch):
        # all are 3D tensor (T+1,B,dim) (Including dummy step at t = -1)
        actions, rewards, terms = batch["act"], batch["rew"], batch["term"]

        # For discrete action space, convert to one-hot vectors.
        # For continuous SAC, keep raw actions.
        if not self.algo.continuous_action:
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T+1, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T+1, B, dim)

        # extend observs, from len = T+1 to len = T+2
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+2, B, dim)

        outputs = self.forward(actions, rewards, observs, terms, masks)
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
        initial=False
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
        )

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
