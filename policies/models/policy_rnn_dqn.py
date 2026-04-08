import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from policies.rl import RL_ALGORITHMS
from policies.models.recurrent_head import RNN_head
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_DQN_RNN(nn.Module):

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        # Shared RNN encoder
        self.head = RNN_head(obs_dim, action_dim, config_seq)
        self.head_target = deepcopy(self.head)

        # Q-value network
        self.qf = self.algo.build_critic(
            input_size=self.head.embedding_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
        )
        self.qf_target = deepcopy(self.qf)

        # Optimizer
        self.critic_optimizer = AdamW(
            [*self.head.parameters(), *self.qf.parameters()],
            lr=config_rl.critic_lr,
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
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        prev_reward = prev_reward.unsqueeze(0)  # (1, B, 1)
        prev_obs = prev_obs.unsqueeze(0)        # (1, B, dim)
        obs = obs.unsqueeze(0)                  # (1, B, dim)

        joint_embed, current_internal_state = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial,
        )

        current_action = self.algo.select_action(
            qf=self.qf,
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state

    def forward(self, actions, rewards, observs, terms, masks):
        """
        actions[t] = a_{t-1}, shape (T+1, B, A)   one-hot
        rewards[t] = r_{t-1}, shape (T+1, B, 1)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        terms[t]   = done_{t-1}, shape (T+1, B, 1)
        masks[t]   = mask_{t-1}, shape (T+1, B, 1)
        """
        assert actions.dim() == rewards.dim() == terms.dim() == observs.dim() == masks.dim() == 3
        assert actions.shape[0] == rewards.shape[0] == terms.shape[0] == observs.shape[0] - 1 == masks.shape[0]

        ### 1. Compute embeddings once
        joint_embeds, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs
        )  # (T+2, B, dim)

        with torch.no_grad():
            target_joint_embeds, _ = self.head_target.forward(
                actions=actions, rewards=rewards, observs=observs
            )  # (T+2, B, dim)

        ### 2. Critic loss (DDQN)
        # Current Q values (with grad) — .detach() used for target computation below
        q_pred_all = self.qf(joint_embeds)  # (T+2, B, A)

        with torch.no_grad():
            # DDQN: online net selects next action, target net evaluates its value
            next_actions = torch.argmax(q_pred_all.detach(), dim=-1, keepdim=True)[1:]  # (T+1, B, 1)
            next_q_target = self.qf_target(target_joint_embeds)[1:]  # (T+1, B, A)
            next_q = next_q_target.gather(-1, next_actions)  # (T+1, B, 1)
            q_target = rewards + (1.0 - terms) * self.gamma * next_q  # (T+1, B, 1)

        # Gather Q(h_t, a_t) from (T+1) slice
        actions_idx = torch.argmax(actions, dim=-1, keepdim=True)  # (T+1, B, 1)
        q_pred = q_pred_all[:-1].gather(-1, actions_idx)  # (T+1, B, 1)

        # Masked Bellman error
        q_pred = q_pred * masks
        q_target = q_target * masks
        qf_loss = F.huber_loss(q_pred, q_target, reduction="none").mean(dim=(1, 2))  # (T+1,)
        critic_loss = qf_loss.mean()

        num_valid = torch.clamp(masks.sum(), min=1.0)
        outputs = {
            "critic_loss": critic_loss.item(),
            "qf_loss": qf_loss.detach(),
            "q": (q_pred.sum() / num_valid).item(),
            "target_q": (q_target.sum() / num_valid).item(),
        }
        outputs.update(d_forward)

        ### 3. Update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                [*self.head.parameters(), *self.qf.parameters()],
                self.clip_grad_norm,
            )
            total_norm = float(grad_norm)
            max_norm = float(self.clip_grad_norm)
            grad_clip_coef = min(1.0, max_norm / (total_norm + 1e-12))
            outputs["raw_grad_norm"] = total_norm
            outputs["grad_clip_coef"] = grad_clip_coef
            outputs["clip_grad_norm"] = self.clip_grad_norm

        self.critic_optimizer.step()

        ### 4. Soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.head, self.head_target, self.tau)
        ptu.soft_update_from_to(self.qf, self.qf_target, self.tau)

    def update(self, batch):
        actions, rewards, terms = batch["act"], batch["rew"], batch["term"]

        actions = F.one_hot(
            actions.squeeze(-1).long(), num_classes=self.action_dim
        ).float()  # (T+1, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T+1, B, dim)

        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+2, B, dim)

        return self.forward(actions, rewards, observs, terms, masks)
