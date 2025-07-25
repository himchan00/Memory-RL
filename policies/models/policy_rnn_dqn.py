import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_critic import Critic_RNN


class ModelFreeOffPolicy_DQN_RNN(nn.Module):
    ARCH = "memory"
    Markov_Actor = False
    Markov_Critic = False

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        # pixel obs
        image_encoder_fn=lambda: None,
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
            config_seq.model,
            config_rl.config_critic,
            self.algo,
            image_encoder=image_encoder_fn(),  # separate weight
        )
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config_rl.critic_lr)
        # target networks
        self.critic_target = deepcopy(self.critic)

    @torch.no_grad()
    def get_initial_info(self, *args, **kwargs):
        return self.critic.get_initial_info(*args, **kwargs)

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
            initial=initial
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

        ### 1. Critic loss
        q_pred, q_target = self.algo.critic_loss(
            markov_critic=self.Markov_Critic,
            critic=self.critic,
            critic_target=self.critic_target,
            observs=observs,
            actions=actions,
            rewards=rewards,
            terms=terms,
            gamma=self.gamma,
        )

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
        }

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad_norm
            )
            outputs["raw_critic_grad_norm"] = grad_norm.item()

        self.critic_optimizer.step()

        ### 3. soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.critic, self.critic_target, self.tau)

    def report_grad_norm(self):
        # may add qf1, policy, etc.
        return {
            "critic_grad_norm": utl.get_grad_norm(self.critic),
            "critic_seq_grad_norm": utl.get_grad_norm(self.critic.seq_model),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, terms = batch["act"], batch["rew"], batch["term"]
        _, batch_size, _ = actions.shape
        if not self.algo.continuous_action:
            # for discrete action space, convert to one-hot vectors
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)

        outputs = self.forward(actions, rewards, observs, terms, masks)
        return outputs
