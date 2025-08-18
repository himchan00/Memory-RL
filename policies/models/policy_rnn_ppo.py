import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu


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
        # pixel obs
        image_encoder_fn=lambda: None,
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

        self.obs_shortcut = config_seq.model.obs_shortcut
        self.full_transition = config_seq.model.full_transition
        self.hyp_emb = config_seq.model.hyp_emb if hasattr(config_seq, "hyp_emb") else False

        ### Build Model
        ## 1. embed action, state, reward (Feed-forward layers first)
        if image_encoder_fn() is None:
            observ_embedding_size = config_seq.model.observ_embedder.hidden_size
            self.observ_embedder = utl.FeatureExtractor(
                obs_dim, observ_embedding_size, F.relu
            )
        else:  # for pixel observation, use external encoder
            self.observ_embedder = image_encoder_fn()
            observ_embedding_size = self.observ_embedder.embedding_size  # reset it

        self.action_embedder = utl.FeatureExtractor(
            action_dim, config_seq.model.action_embedder.hidden_size, F.relu
        )
        self.reward_embedder = utl.FeatureExtractor(
            1, config_seq.model.reward_embedder.hidden_size, F.relu
        )

        ## 2. build RNN model
        observ_hidden_size = 2 * observ_embedding_size if self.full_transition else observ_embedding_size
        rnn_input_size = (
            observ_hidden_size
            + config_seq.model.action_embedder.hidden_size
            + config_seq.model.reward_embedder.hidden_size
        )
        self.seq_model = SEQ_MODELS[config_seq.model.seq_model_config.name](
            input_size=rnn_input_size, **config_seq.model.seq_model_config.to_dict()
        )

        ## 3. build actor-critic
        # q-value networks
        self.hidden_dim = self.seq_model.hidden_size
        if self.seq_model.name == "hist":
            if self.seq_model.agg == "mean" and self.seq_model.temb_mode == "concat":
                self.hidden_dim += config_seq.model.seq_model_config.temb_size
        input_size = self.hidden_dim
        if self.obs_shortcut:
            input_size += observ_embedding_size

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
            *self.observ_embedder.parameters(),
            *self.action_embedder.parameters(),
            *self.reward_embedder.parameters(),
            *self.seq_model.parameters(),
            *self.critic.parameters(),
            *self.policy.parameters(),
        ]

    def get_hidden_states(
        self, actions, rewards, observs, initial_internal_state=None
    ):
        """
        Inputs:
        actions[t] = a_t, shape (L, B, dim)
        rewards[t] = r_t, shape (L, B, dim)
        observs[t] = o_t, shape (L+1, B, dim)
        Outputs:
        hidden[t] = h_t: (L, B, dim)
        """
        observs_t = observs[:-1] # o[t]
        observs_t_1 = observs[1:] # o[t+1]
        input_a = self.action_embedder(actions)
        input_r = self.reward_embedder(rewards)
        if self.full_transition:
            input_s = self.observ_embedder(observs_t)
            input_s_1 = self.observ_embedder(observs_t_1)
            inputs = torch.cat((input_a, input_r, input_s, input_s_1), dim=-1)
        elif self.obs_shortcut:
            input_s = self.observ_embedder(observs_t)
            inputs = torch.cat((input_a, input_r, input_s), dim=-1)
        else: 
            input_s_1 = self.observ_embedder(observs_t_1)
            inputs = torch.cat((input_a, input_r, input_s_1), dim=-1)

        if initial_internal_state is None:  # training
            initial_internal_state = self.seq_model.get_zero_internal_state(
                batch_size=inputs.shape[1]
            )  # initial_internal_state is zeros
            output, _ = self.seq_model(inputs, initial_internal_state)
            return output
        else:  # useful for one-step rollout
            output, current_internal_state = self.seq_model(
                inputs, initial_internal_state
            )
            return output, current_internal_state

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
        bs = actions.shape[1]

        if not self.obs_shortcut:
            o, a, r = ptu.zeros((1, bs, self.obs_dim)).float(), ptu.zeros((1, bs, self.action_dim)).float(), ptu.zeros((1, bs, 1)).float() 
            in_observs, in_actions, in_rewards = torch.cat((o, observs), dim = 0), torch.cat((a, actions[:-1]), dim = 0), torch.cat((r, rewards), dim = 0)
        else:
            in_observs, in_actions, in_rewards = observs, actions[:-1], rewards
        hidden_states = self.get_hidden_states(
            actions=in_actions, rewards=in_rewards, observs=in_observs
        )
        
        if self.obs_shortcut:
            h = self.get_initial_hidden(bs)
            hidden_states = torch.cat((h, hidden_states), dim = 0)

        if self.hyp_emb:
            rms = torch.mean(hidden_states ** 2, dim = -1, keepdim = True) ** 0.5 
            hidden_embeds = hidden_states * torch.tanh(rms)/rms.clamp(min=1e-6) # avoid division by zero
        else:
            hidden_embeds = hidden_states # (T+1, B, dim)

        if self.obs_shortcut:
            observs_embeds = self.observ_embedder(observs) # Recomputing observes_embed is not computationally efficient. Modification required.
            joint_embeds = torch.cat((observs_embeds, hidden_embeds), dim = -1) # Q(s, h)
        else:
            joint_embeds = hidden_embeds # Q(h)

        # NOTE: When time embedding information is concatenated to the hidden state, the resulting hidden state dimension can be larger than hidden_size.
        d_forward = {"hidden_states_mean": hidden_states[:, :, :self.seq_model.hidden_size].detach().mean(dim = (1, 2)),
                     "hidden_states_std": hidden_states[:, :, :self.seq_model.hidden_size].detach().std(dim = 2).mean(dim = 1)}
        if self.hyp_emb:
            d_forward["hidden_embeds_mean"] = hidden_embeds.detach().mean(dim = (1, 2))
            d_forward["hidden_embeds_std"] = hidden_embeds.detach().std(dim = 2).mean(dim = 1)
        if self.obs_shortcut:
            d_forward["observs_embeds_mean"] = observs_embeds.detach().mean(dim = (1, 2))
            d_forward["observs_embeds_std"] = observs_embeds.detach().std(dim = 2).mean(dim = 1)


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
    def get_initial_info(self, max_attn_span: int = -1):
        prev_obs = ptu.zeros((1, self.obs_dim)).float()
        prev_action = ptu.zeros((1, self.action_dim)).float()
        reward = ptu.zeros((1, 1)).float()
        internal_state = self.seq_model.get_zero_internal_state()

        return prev_obs, prev_action, reward, internal_state

    @torch.no_grad()
    def get_initial_hidden(self, batch_size):
        if self.seq_model.name == "hist":
            h = self.seq_model.get_zero_hidden_state(batch_size = batch_size)
        else:
            h = ptu.zeros((1, batch_size, self.hidden_dim)).float()
        return h
    
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

        if initial and self.obs_shortcut:
            hidden_state = self.get_initial_hidden(batch_size = 1).squeeze(0) # (1. hidden_dim)
            current_internal_state = self.seq_model.get_zero_internal_state()
        else:
            if initial and not self.obs_shortcut:
                prev_obs, prev_action, prev_reward = ptu.zeros((1, 1, self.obs_dim)).float(), ptu.zeros((1, 1, self.action_dim)).float(), ptu.zeros((1, 1, 1)).float() 
                prev_internal_state = self.seq_model.get_zero_internal_state()
                
            hidden_state, current_internal_state = self.get_hidden_states(
                actions=prev_action,
                rewards=prev_reward,
                observs=torch.cat((prev_obs, obs), dim = 0),
                initial_internal_state=prev_internal_state,
            )
            hidden_state = hidden_state.squeeze(0)  # (B, dim)

        if self.hyp_emb:
            rms = torch.mean(hidden_state ** 2, dim = -1, keepdim = True) ** 0.5 
            hidden_embed = hidden_state * torch.tanh(rms)/rms.clamp(min=1e-6) # avoid division by zero
        else:
            hidden_embed = hidden_state

        if self.obs_shortcut:
            obs_embed = self.observ_embedder(obs) # Recomputing observes_embed is not computationally efficient. Modification required.
            joint_embed = torch.cat((obs_embed.squeeze(0), hidden_embed), dim = -1)
        else:
            joint_embed = hidden_embed

        
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

