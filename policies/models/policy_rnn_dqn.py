import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic import Critic_RNN
import math


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

        # Critics
        self.critic = Critic_RNN(
            obs_dim,
            action_dim,
            config_seq,
            config_rl.config_critic,
            self.algo,
        )

        # target networks
        self.critic_target = deepcopy(self.critic)
        self.transition_dropout = 0.0
        if self.critic.head.seq_model.name == "hist":
            self.critic.head.seq_model.is_target = False
            self.critic_target.head.seq_model.is_target = True

        # optimizer
        self.critic_optimizer = AdamW(self.critic.parameters(), lr=config_rl.critic_lr)


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

        current_action, current_internal_state = self.critic.act(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            deterministic=deterministic,
            initial=initial,
        )

        return current_action, current_internal_state

    def forward(self, actions, rewards, observs, terms, masks):
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
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        if self.transition_dropout > 0.0:
            mask = self.critic.head.seq_model.sample_transition_dropout_mask(length=len(actions)-1, p=self.transition_dropout)
            self.critic.head.seq_model.transition_dropout_mask = mask
            self.critic_target.head.seq_model.transition_dropout_mask = mask
        ### 1. Critic loss
        q_pred, q_target, d_loss = self.algo.critic_loss(
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            terms=terms,
            gamma=self.gamma,
        )
        if self.transition_dropout > 0.0:
            self.critic.head.seq_model.transition_dropout_mask = None
            self.critic_target.head.seq_model.transition_dropout_mask = None

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        # this is not equal to masks * q1_pred, cuz the denominator in mean()
        # 	should depend on masks > 0.0, not a constant B*T
        q_pred = q_pred * masks
        q_target = q_target * masks
        qf_loss = ((q_pred - q_target) ** 2).sum() / num_valid  # TD error

        self.critic_optimizer.zero_grad()
        qf_loss.backward()

        outputs = {
            "critic_loss": qf_loss.item(),
            "q": (q_pred.sum() / num_valid).item(),
            "target_q": (q_target.sum() / num_valid).item(),
        }
        outputs.update(d_loss)

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad_norm
            )
            total_norm = float(grad_norm)
            max_norm = float(self.clip_grad_norm)
            grad_clip_coef = min(1.0, max_norm / (total_norm + 1e-12))
            outputs["raw_grad_norm"] = total_norm
            outputs["grad_clip_coef"] = grad_clip_coef
            outputs["clip_grad_norm"] = self.clip_grad_norm


        self.critic_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)


    def update(self, batch):
        # all are 3D tensor (T+1,B,dim) (Including dummy step at t = -1)
        actions, rewards, terms = batch["act"], batch["rew"], batch["term"]

        # for discrete action space, convert to one-hot vectors
        actions = F.one_hot(
            actions.squeeze(-1).long(), num_classes=self.action_dim
        ).float()  # (T+1, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T+1, B, dim)

        # extend observs, from len = T+1 to len = T+2
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+2, B, dim)

        outputs = self.forward(actions, rewards, observs, terms, masks)
        return outputs
