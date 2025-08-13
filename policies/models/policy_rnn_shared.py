import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from utils import helpers as utl
from policies.rl import RL_ALGORITHMS
from policies.seq_models import SEQ_MODELS
import torchkit.pytorch_utils as ptu


class ModelFreeOffPolicy_Shared_RNN(nn.Module):
    """
    Recurrent Actor and Recurrent Critic with shared RNN
    We find `freeze_critic = True` can prevent degradation shown in https://github.com/twni2016/pomdp-baselines
    """

    ARCH = "memory"

    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        freeze_critic: bool,
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

        self.freeze_critic = freeze_critic

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

        # NOTE: For continuous SAC, algo.build_critic will internally expect [input_size + action_dim].
        # For discrete SACD, it returns (A)-dim outputs from just [input_size].
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=input_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=action_dim,
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

        # policy network
        self.policy = self.algo.build_actor(
            input_size=input_size,
            action_dim=self.action_dim,
            hidden_sizes=config_rl.config_actor.hidden_dims,
        )
        # target networks
        self.policy_target = deepcopy(self.policy)

        # use joint optimizer
        assert config_rl.critic_lr == config_rl.actor_lr
        self.optimizer = Adam(self._get_parameters(), lr=config_rl.critic_lr)

    def _get_parameters(self):
        # exclude targets
        return [
            *self.observ_embedder.parameters(),
            *self.action_embedder.parameters(),
            *self.reward_embedder.parameters(),
            *self.seq_model.parameters(),
            *self.qf1.parameters(),
            *self.qf2.parameters(),
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

    def forward(self, actions, rewards, observs, terms, masks):
        """
        actions[t] = a_t, shape (T, B, dim)
        rewards[t] = r_t, shape (T, B, dim)
        observs[t] = o_t, shape (T+1, B, dim)
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
        bs = actions.shape[1]
        num_valid = torch.clamp(masks.sum(), min=1.0)  # as denominator of loss

        if not self.obs_shortcut:
            o, a, r = ptu.zeros((1, bs, self.obs_dim)).float(), ptu.zeros((1, bs, self.action_dim)).float(), ptu.zeros((1, bs, 1)).float() 
            in_observs, in_actions, in_rewards = torch.cat((o, observs), dim = 0), torch.cat((a, actions), dim = 0), torch.cat((r, rewards), dim = 0)
        else:
            in_observs, in_actions, in_rewards = observs, actions, rewards
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

        ### 2. Critic loss

        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            # first next_actions from target/current policy, (T+1, B, dim) including reaction to last obs
            # new_next_actions: (T+1, B, dim for continuous) or (T+1, B, A for discrete probs)
            # new_next_log_probs: (T+1, B, 1) for continuous OR (T+1, B, A) for discrete
            new_next_actions, new_next_log_probs = self.algo.forward_actor_in_target(
                actor=self.policy,
                actor_target=self.policy_target,
                next_observ=joint_embeds,
            )

            if self.algo.continuous_action:
                target_joint_embeds = torch.cat((joint_embeds, new_next_actions), dim = -1)
            next_q1 = self.qf1_target(target_joint_embeds) # (T+1,B,1) if cont_act else (T+1,B,A)
            next_q2 = self.qf2_target(target_joint_embeds)
            min_next_q_target = torch.min(next_q1, next_q2) # (T+1,B,1) if cont_act else (T+1,B,A)
            min_next_q_target += self.algo.entropy_bonus(new_next_log_probs)
            if not self.algo.continuous_action:
                min_next_q_target = (new_next_actions * min_next_q_target).sum(dim=-1, keepdims=True)  
            min_next_q_target = min_next_q_target[1:]  # (T,B,1)
            q_target = rewards + (1.0 - terms) * self.gamma * min_next_q_target

        # Q(h(t), a(t)) (T, B, 1)
        # 3. joint embeds
        curr_joint_embeds = joint_embeds[:-1]
        if self.algo.continuous_action: # Continuous: Q(h_t, a_t) using stored actions (T,B,act_dim)
            curr_joint_embeds = torch.cat((curr_joint_embeds, actions), dim = -1)
        q1_pred = self.qf1(curr_joint_embeds)
        q2_pred = self.qf2(curr_joint_embeds)
        if not self.algo.continuous_action: # Discrete (original): gather on action id from logits (T,B,A)->(T,B,1)
            actions_idx = torch.argmax(actions, dim=-1, keepdims=True)  # (T,B,1)
            q1_pred = q1_pred.gather(dim=-1, index=actions_idx)  # (T,B,1)
            q2_pred = q2_pred.gather(dim=-1, index=actions_idx)  # (T,B,1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid  # TD error
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid  # TD error

        ### 3. Actor loss
        # Continuous: J_pi = E[ alpha*logpi - minQ ]
        # Discrete:   E_{a~pi}[ Q + H ]
        new_actions, new_log_probs = self.algo.forward_actor(
            actor=self.policy, observ=joint_embeds
        )

        if self.freeze_critic:
            ######## freeze critic parameters
            ######## and detach critic hidden states
            if self.algo.continuous_action:
                new_joint_embeds = torch.cat((joint_embeds.detach(), new_actions), dim = -1) # (T+1, B, dim)
            else:
                new_joint_embeds = joint_embeds.detach()

            freezed_qf1 = deepcopy(self.qf1).to(ptu.device)
            freezed_qf2 = deepcopy(self.qf2).to(ptu.device)
            q1 = freezed_qf1(new_joint_embeds)
            q2 = freezed_qf2(new_joint_embeds)
        else:
            if self.algo.continuous_action:
                new_joint_embeds = torch.cat((joint_embeds, new_actions), dim = -1) # (T+1, B, dim)
            else:
                new_joint_embeds = joint_embeds

            q1 = self.qf1(new_joint_embeds)
            q2 = self.qf2(new_joint_embeds)

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1) or (T+1,B,A)
        policy_loss = -min_q_new_actions
        policy_loss += -self.algo.entropy_bonus(new_log_probs)

        if not self.algo.continuous_action:
            policy_loss = (new_actions * policy_loss).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)
            new_log_probs = (new_actions * new_log_probs).sum(
                axis=-1, keepdims=True
            )  # (T+1,B,1)

        policy_loss = policy_loss[:-1]  # (T,B,1) remove the last obs
        policy_loss = (policy_loss * masks).sum() / num_valid

        ### 4. update
        total_loss = 0.5 * (qf1_loss + qf2_loss) + policy_loss

        outputs = {
            "critic_loss": (qf1_loss + qf2_loss).item(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
            "actor_loss": policy_loss.item(),
        }
        outputs.update(d_forward)

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm
            )
            outputs["raw_grad_norm"] = grad_norm.item()

        self.optimizer.step()

        ### 5. soft update
        self.soft_target_update()

        ### 6. update others like alpha
        if new_log_probs is not None:
            # extract valid log_probs
            with torch.no_grad():
                current_log_probs = (new_log_probs[:-1] * masks).sum() / num_valid
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.use_target_actor:
            ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def report_grad_norm(self):
        return {
            "seq_grad_norm": utl.get_grad_norm(self.seq_model),
            "critic_grad_norm": utl.get_grad_norm(self.qf1),
            "actor_grad_norm": utl.get_grad_norm(self.policy),
        }

    def update(self, batch):
        # all are 3D tensor (T,B,dim)
        actions, rewards, terms = batch["act"], batch["rew"], batch["term"]

        # For discrete action space, convert to one-hot vectors.
        # For continuous SAC, keep raw actions.
        if not self.algo.continuous_action:
            actions = F.one_hot(
                actions.squeeze(-1).long(), num_classes=self.action_dim
            ).float()  # (T, B, A)

        masks = batch["mask"]
        obs, next_obs = batch["obs"], batch["obs2"]  # (T, B, dim)

        # extend observs, from len = T to len = T+1
        observs = torch.cat((obs[[0]], next_obs), dim=0)  # (T+1, B, dim)

        outputs = self.forward(actions, rewards, observs, terms, masks)
        return outputs

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
        initial=False
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

        # 4. Actor head, generate action tuple
        current_action = self.algo.select_action(
            actor=self.policy,
            observ=joint_embed,
            deterministic=deterministic,
        )

        return current_action, current_internal_state
