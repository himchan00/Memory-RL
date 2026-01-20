import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from policies.rl import RL_ALGORITHMS
from policies.models.recurrent_head import RNN_head
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp
from utils.helpers import get_constant_schedule_with_warmup


class ModelFreeOffPolicy_Shared_RNN(nn.Module):
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
        self.head_target = deepcopy(self.head)

        self.permutation_training = config_seq.get("permutation_training", False)
        if self.head.seq_model.name == "mate":
            self.head.is_target = False
            self.head_target.is_target = True
            print(f"Use permutation training: {self.permutation_training}")

        if self.algo.continuous_action:
            # action embedder for continuous action space
            # NOTE: This is not used in discrete action space since we can directly use one-hot encoding.
            # NOTE: This is different from the action embedding in RNN_head, which is for both discrete and continuous action space.
            self.action_embedder = Mlp(
                input_size=action_dim,
                output_size=4*action_dim, # expand dimension for better representation
                **config_seq.embedder.to_dict(),
            )
            # target networks
            self.action_embedder_target = deepcopy(self.action_embedder)

        # q-value networks
        # NOTE: For continuous SAC, algo.build_critic will internally expect [input_size + action_dim].
        # For discrete SACD, it returns (A)-dim outputs from just [input_size].
        self.qf1, self.qf2 = self.algo.build_critic(
            input_size=self.head.embedding_size,
            hidden_sizes=config_rl.config_critic.hidden_dims,
            action_dim=self.action_embedder.output_size if self.algo.continuous_action else action_dim,
        )
        # target networks
        self.qf1_target = deepcopy(self.qf1)
        self.qf2_target = deepcopy(self.qf2)

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
        if self.algo.continuous_action:
            params += [
                *self.action_embedder.parameters(),
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

        if self.permutation_training:
            length, batch_size, _ = actions.shape
            transition_perm, memory_perm = self.head.seq_model.sample_permutation_indices(length-1, batch_size)
            self.head.transition_perm = self.head_target.transition_perm = transition_perm
            self.head.memory_perm = self.head_target.memory_perm = memory_perm

        joint_embeds, d_forward = self.head.forward(actions=actions, rewards=rewards, observs=observs)
        with torch.no_grad():
            target_joint_embeds, _ = self.head_target.forward(actions=actions, rewards=rewards, observs=observs)

        if self.permutation_training:
            self.head.transition_perm = self.head_target.transition_perm = None
            self.head.memory_perm = self.head_target.memory_perm = None

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
                new_next_action_embeds = self.action_embedder_target(new_next_actions) # (T+1, B, dim)
                target_joint_embeds = torch.cat((target_joint_embeds, new_next_action_embeds), dim = -1)
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
        if self.algo.continuous_action: # Continuous: Q(h_t, a_t) using stored actions (T,B,act_dim)
            action_embeds = self.action_embedder(actions) # (T+1, B, dim)
            curr_joint_embeds = torch.cat((joint_embeds[:-1], action_embeds), dim = -1)
        else:
            curr_joint_embeds = joint_embeds[:-1]

        q1_pred = self.qf1(curr_joint_embeds)
        q2_pred = self.qf2(curr_joint_embeds)
        if not self.algo.continuous_action: # Discrete (original): gather on action id from logits (T,B,A)->(T,B,1)
            actions_idx = torch.argmax(actions, dim=-1, keepdims=True)  # (T,B,1)
            q1_pred = q1_pred.gather(dim=-1, index=actions_idx)  # (T,B,1)
            q2_pred = q2_pred.gather(dim=-1, index=actions_idx)  # (T,B,1)

        # masked Bellman error: masks (T,B,1) ignore the invalid error
        q1_pred, q2_pred = q1_pred * masks, q2_pred * masks
        q_target = q_target * masks

        qf1_loss = torch.nn.HuberLoss(reduction='none')(q1_pred, q_target).mean(dim=(1, 2))
        qf2_loss = torch.nn.HuberLoss(reduction='none')(q2_pred, q_target).mean(dim=(1, 2))

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
            new_action_embeds = self.action_embedder(new_actions) # (T+1, B, dim)
            new_joint_embeds = torch.cat((joint_embeds, new_action_embeds), dim = -1) # (T+1, B, dim)
        else:
            new_joint_embeds = joint_embeds

        q1 = self.qf1(new_joint_embeds)
        q2 = self.qf2(new_joint_embeds)
        if self.freeze_critic:
            self._unfreeze_critic()

        min_q_new_actions = torch.min(q1, q2)  # (T+1,B,1) or (T+1,B,A)
        policy_loss = -min_q_new_actions
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
        outputs = {
            "critic_loss": qf_loss.mean().item(),
            "qf_loss": qf_loss.detach(),
            "q1": (q1_pred.sum() / num_valid).item(),
            "q2": (q2_pred.sum() / num_valid).item(),
            "actor_loss": policy_loss.mean().item(),
            "policy_loss": policy_loss.detach(),

        }
        outputs.update(d_forward)

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_norm = nn.utils.clip_grad_norm_(
                self._get_parameters(), self.clip_grad_norm # Only clip gradients of the RNN head.
            )
            total_norm = float(grad_norm)
            max_norm = float(self.clip_grad_norm)
            grad_clip_coef = min(1.0, max_norm / (total_norm + 1e-12))
            outputs["raw_grad_norm"] = total_norm
            outputs["grad_clip_coef"] = grad_clip_coef
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
                current_log_probs = current_log_probs.item()

            other_info = self.algo.update_others(current_log_probs)
            outputs.update(other_info)
        
        return outputs

    
    def _freeze_critic(self):
        for param in self.qf1.parameters():
            param.requires_grad = False
        for param in self.qf2.parameters():
            param.requires_grad = False
        if self.algo.continuous_action:
            for param in self.action_embedder.parameters():
                param.requires_grad = False

    def _unfreeze_critic(self):
        for param in self.qf1.parameters():
            param.requires_grad = True
        for param in self.qf2.parameters():
            param.requires_grad = True
        if self.algo.continuous_action:
            for param in self.action_embedder.parameters():
                param.requires_grad = True

    def soft_target_update(self):
        ptu.soft_update_from_to(self.head, self.head_target, self.tau)
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        if self.algo.continuous_action:
            ptu.soft_update_from_to(self.action_embedder, self.action_embedder_target, self.tau)
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
