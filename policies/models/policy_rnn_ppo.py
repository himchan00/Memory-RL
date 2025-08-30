import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from policies.rl import RL_ALGORITHMS
import torchkit.pytorch_utils as ptu
from policies.models.recurrent_head import RNN_head


class ModelFreePPO_Shared_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    """

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
        self.lam = config_rl.lam
        self.ppo_epochs = config_rl.ppo_epochs
        self.eps_clip = config_rl.eps_clip
        self.ent_coef = config_rl.ent_coef
        self.vf_coef = config_rl.vf_coef
        self.normalize_advantage = config_rl.normalize_advantage
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm

        self.algo = RL_ALGORITHMS[config_rl.algo](
            action_dim=action_dim, **config_rl.to_dict()
        )

        self.head = RNN_head(
            obs_dim,
            action_dim,
            config_seq,
        )
        ## 3. build actor-critic
        # q-value networks
        input_size = self.head.embedding_size
        self.critic = self.algo.build_critic(
            input_size=input_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
        )

        # policy network
        self.policy = self.algo.build_actor(
            input_size=input_size,
            hidden_sizes=config_rl.config_actor.hidden_dims,
            action_dim=self.action_dim,
            continuous_action=self.algo.continuous_action
        )

        # use joint optimizer
        self.optimizer = Adam(self._get_parameters(), lr=config_rl.lr)

    def _get_parameters(self):
        # exclude targets
        return [
            *self.head.parameters(),
            *self.critic.parameters(),
            *self.policy.parameters(),
        ]


    def forward(self, actions, rewards, observs):
        """
        actions[t] = a_t, shape (T, B, dim)
        rewards[t] = r_t, shape (T-1, B, dim)
        observs[t] = o_t, shape (T, B, dim)
        """
        assert (
            actions.dim()
            == rewards.dim()
            == observs.dim()
            == 3
        )
        assert (
            actions.shape[0]
            == rewards.shape[0] + 1
            == observs.shape[0]
        )
        joint_embeds, d_forward = self.head.forward(actions=actions[:-1], rewards=rewards, observs=observs)

        if not self.algo.continuous_action:
            actions = actions.argmax(dim = -1) 

        current_log_prob = self.policy(obs=joint_embeds, return_log_prob=True, action=actions)[-1] 
        current_v = self.critic(joint_embeds)

        return current_log_prob, current_v, d_forward


    def update(self, buffer):
        buffer.compute_gae(gamma = self.gamma, lam = self.lam)
        # all are 3D tensor (T,B,dim)
        actions, rewards = buffer.actions, buffer.rewards
        obs, next_obs = buffer.observations, buffer.next_observations
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)
        logprobs, returns, advantages, masks = buffer.logprobs.squeeze(-1), buffer.returns.squeeze(-1), buffer.advantages.squeeze(-1), buffer.masks.squeeze(-1) # (T, B)
        # For discrete action space, convert to one-hot vectors.
        if not self.algo.continuous_action:
            actions = F.one_hot(actions.squeeze(-1).long(), num_classes=self.action_dim).float()  # (T, B, A)

        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_loss_sum = 0
        grad_norm_sum = 0
        d_forward_sum = {}
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss
        for _ in range(self.ppo_epochs):
            # Evaluating old actions and values
            new_logprobs, new_values, d_forward = self.forward(actions, rewards[:-1], observs[:-1])
            # match dimensions with advantages tensor
            new_logprobs = new_logprobs.squeeze(-1) # (T, B)
            new_values = new_values.squeeze(-1) # (T, B)
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - logprobs) # (T, B)
            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # final loss of clipped objective PPO
            policy_loss = -(masks * torch.min(surr1, surr2)).sum()/num_valid
            value_loss = (masks*((new_values - returns) ** 2)).sum()/num_valid
            entropy_loss = (masks * new_logprobs).sum()/num_valid
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip and self.clip_grad_norm > 0.0:
                grad_norm = nn.utils.clip_grad_norm_(
                    self._get_parameters(), self.clip_grad_norm
                )
                grad_norm_sum += grad_norm
            self.optimizer.step()

            loss_sum += loss.detach()
            policy_loss_sum += policy_loss.detach()
            value_loss_sum += value_loss.detach()
            entropy_loss_sum += entropy_loss.detach()
            for k, v in d_forward.items():
                if k not in d_forward_sum:
                    d_forward_sum[k] = v
                else:
                    d_forward_sum[k] += v

        # reset buffer
        buffer.reset()

        d_update = {"loss":(loss_sum/self.ppo_epochs).item() ,"policy_loss": (policy_loss_sum/self.ppo_epochs).item(), "value_loss": (value_loss_sum/self.ppo_epochs).item(),
                 "entropy_loss": (entropy_loss_sum/self.ppo_epochs).item(), "raw_grad_norm": (grad_norm_sum/self.ppo_epochs).item()}

        for k, v in d_forward_sum.items():
            d_update[k] = v / self.ppo_epochs
        return d_update

    @torch.no_grad()
    def get_initial_info(self, batch_size):
        prev_obs = ptu.zeros((batch_size, self.obs_dim)).float()
        prev_action = ptu.zeros((batch_size, self.action_dim)).float()
        reward = ptu.zeros((batch_size, 1)).float()
        internal_state = self.head.seq_model.get_zero_internal_state(batch_size=batch_size)

        return prev_obs, prev_action, reward, internal_state

    
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
        return_logprob_v=False,
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
            initial=initial
        )        
        
        if return_logprob_v:
            output = self.policy(obs = joint_embed, deterministic = deterministic, return_log_prob = True)
            current_action, current_log_prob = output[0], output[-1]
            if not self.algo.continuous_action:
                action_indices = current_action.argmax(dim=-1, keepdim=True)  # (B, 1)
                current_log_prob = current_log_prob.gather(1, action_indices)  # (B, 1)
            return current_action, current_internal_state, current_log_prob, self.critic(joint_embed)
        else:
            current_action = self.policy(obs = joint_embed, deterministic = deterministic)[0]
            return current_action, current_internal_state

