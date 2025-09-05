import time

import math
import numpy as np
import torch
from torch.nn import functional as F

from .models import AGENT_CLASSES

# For PPO
from buffers.rollout_buffer import RolloutBuffer

from torchkit import pytorch_utils as ptu
import matplotlib.pyplot as plt
import wandb


class Learner:
    def __init__(self, env, eval_env, FLAGS, config_rl, config_seq, config_env):
        self.train_env = env
        self.eval_env = eval_env
        self.FLAGS = FLAGS
        self.config_rl = config_rl
        self.config_seq = config_seq
        self.config_env = config_env

        self.init_env()

        self.init_agent()

        self.init_train()

    def init_env(
        self,
    ):
        # get action / observation dimensions
        action_space = self.train_env.get_attr("action_space")[0]
        observation_space = self.train_env.get_attr("observation_space")[0]
        if action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = action_space.shape[0]
            self.act_continuous = True
        else:
            assert action_space.__class__.__name__ == "Discrete"
            self.act_dim = action_space.n
            self.act_continuous = False
        self.obs_dim = observation_space.shape[0]
        self.n_env = self.config_env.n_env
        print("obs space", observation_space)
        print("act space", action_space)
        print("obs_dim", self.obs_dim, "act_dim", self.act_dim, "n_env", self.n_env)

    def init_agent(
        self,
    ):
        # initialize agent
        if self.config_rl.algo == 'ppo':
            agent_class = AGENT_CLASSES["Policy_PPO_RNN"]
        elif self.config_rl.algo == "dqn":
            agent_class = AGENT_CLASSES["Policy_DQN_RNN"]
        elif self.config_rl.algo == "sac":
            agent_class = AGENT_CLASSES["Policy_Shared_RNN"]

        self.agent = agent_class(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            config_seq=self.config_seq,
            config_rl=self.config_rl,
            freeze_critic=self.FLAGS.freeze_critic,
        ).to(ptu.device)


    def init_train(
        self,
    ):

        if self.config_rl.algo == "ppo":
            num_episodes = self.FLAGS.batch_size
            is_ppo = True
        else:
            num_episodes = int(self.config_rl.replay_buffer_num_episodes)
            is_ppo = False
        self.policy_storage = RolloutBuffer(observation_dim=self.obs_dim,
                                        action_dim=self.act_dim if self.act_continuous else None,  # save memory
                                        max_episode_len=self.train_env.get_attr("max_episode_steps")[0],
                                        num_episodes=num_episodes,
                                        is_ppo=is_ppo
                                    )

        self.total_episodes = self.FLAGS.start_training + self.FLAGS.train_episodes

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_rl_update_steps_total = 0
        self._n_episodes_total = 0
        self._start_time = time.time()


    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.FLAGS.start_training > 0: # Set to 0 for PPO
            while self._n_episodes_total < self.FLAGS.start_training:
                self.collect_rollouts(num_rollouts=1, random_actions=True)

            self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )

        while self._n_episodes_total < self.total_episodes:
            if self.config_rl.algo == "ppo":
                d_rollout, env_steps = self.collect_rollouts(num_rollouts=self.FLAGS.batch_size // self.n_env)
                d_update = self.update_ppo()
            else:
                d_rollout, env_steps = self.collect_rollouts(num_rollouts=1)
                d_update = self.update(
                    int(math.ceil(self.FLAGS.updates_per_step * env_steps))
                )  # NOTE: ceil to make sure at least 1 step


            if self._n_episodes_total % self.config_env.log_interval == 0:
                # logging
                d_train = {**d_rollout, **d_update}
                visualize = self._n_episodes_total % (self.config_env.visualize_every * self.config_env.log_interval) == 0
                d_train = self.process_and_log_train(d_train, visualize=visualize)
                d_info = {"info/env_steps": self._n_env_steps_total, "info/rl_update_steps": self._n_rl_update_steps_total, \
                            "info/duration_minute": (time.time() - self._start_time)/60}
                wandb.log(d_info, self._n_episodes_total)

            # evaluate and log
            if self._n_episodes_total % self.config_env.eval_interval == 0:
                visualize = self._n_episodes_total % (self.config_env.visualize_every * self.config_env.eval_interval) == 0 and self.config_env.visualize_env
                d_rollout, frames = self.collect_rollouts(num_rollouts=self.config_env.eval_episodes // self.n_env, mode="eval", visualize=visualize, deterministic=True)
                print(f"Total rollouts:{self._n_episodes_total}, Return: {d_rollout['return']:.2f}, Success rate: {d_rollout['success_rate']:.2f}, Episode_len: {d_rollout['episode_len']:.2f}")
                d_eval = {}
                for k, v in d_rollout.items():
                    d_eval["eval/" + k] = v
                wandb.log(d_eval, self._n_episodes_total)
                if frames is not None:
                    wandb.log({"eval/visualization": wandb.Video(np.array(frames).transpose(0,3,1,2), fps=30, format="gif")}, self._n_episodes_total)


    def process_and_log_train(self, d_train, visualize=False):
        """
        Processes and log training data.

        If visualize is True:
            - For (T,) data, generate matplotlib Figure objects for
              metric vs. time (t) plots.
        Scalar metrics are retained as-is (not processed).
        """

        for key, value in d_train.items():
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                if visualize:
                    value = value.cpu().numpy()
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.plot(value)
                    ax.set_title(f"{key} vs t", fontsize = 20)
                    ax.set_xlabel("t", fontsize = 16)
                    ax.set_ylabel(f"{key}", fontsize = 16)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    plt.tight_layout()

                    wandb.log({"visualizations/" + key : wandb.Image(fig)}, self._n_episodes_total)

                    plt.close(fig)
            else:
                # scalar metrics are retained
                wandb.log({"train/" + key : value}, self._n_episodes_total)
        return None



    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False, mode = "train", deterministic = False, visualize=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        mode: param mode: whether to collect rollouts in "train" or "eval" mode
        """
        assert mode in ["train", "eval"]
        if visualize or deterministic:
            assert mode == "eval", "Visualization & Deterministic modes is only supported in eval mode."
        self.agent.eval()  # set to eval mode for deterministic dropout
        before_env_steps = self._n_env_steps_total
        returns_per_episode = np.zeros(num_rollouts)
        success_rate = np.zeros(num_rollouts)
        avg_steps = np.zeros(num_rollouts)
        frames = None
        current_env = self.train_env if mode == "train" else self.eval_env
        for idx in range(num_rollouts):
            steps = 0
            running_rewards = 0.0

            obs = ptu.from_numpy(current_env.reset()[0])  # reset
            obs = obs.reshape(-1, obs.shape[-1]) # (n_env, obs_dim)
            term = ptu.zeros((self.n_env, 1))
            episode_success = ptu.zeros((self.n_env, 1))
            done_rollout = False

            if mode == "train":
                obs_list, act_list, rew_list, next_obs_list, term_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                if self.config_rl.algo == "ppo":
                    logprob_list, value_list = ([], [])
            else: # eval
                if visualize and idx == 0: # Visualization only for the first rollout
                    frames = []

            if not random_actions:
                # Dummy variables, not used
                prev_obs, action, reward, internal_state = self.agent.get_initial_info(batch_size = self.n_env)
                initial=True

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor([current_env.action_space.sample()]).reshape(self.n_env, -1)  # (B, A) for continuous action, (B, 1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.squeeze(-1).long(), num_classes=self.act_dim
                        ).float()  # (B, A)

                else:
                    if self.config_rl.algo == "ppo" and mode == "train":
                        action, internal_state, logprob, value = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            prev_reward=reward,
                            prev_obs=prev_obs,
                            obs=obs,
                            deterministic=False,
                            initial=initial,
                            return_logprob_v=True
                        )
                    else:
                        action, internal_state = self.agent.act(
                            prev_internal_state=internal_state,
                            prev_action=action,
                            prev_reward=reward,
                            prev_obs=prev_obs,
                            obs=obs,
                            deterministic=deterministic,
                            initial=initial,
                        )
                    initial=False

                # Process and validate action
                np_action = ptu.get_numpy(action)
                if not self.act_continuous:
                    np_action = np.argmax(np_action, axis=-1)  # one-hot to int
                if not current_env.action_space.contains(np_action):
                    raise ValueError("Invalid action!")
                # env step
                next_obs, reward, terminated, truncated, info = current_env.step(np_action)
                next_obs = ptu.from_numpy(next_obs).view(self.n_env, -1)
                reward = ptu.FloatTensor(reward).view(self.n_env, -1)  # (B, 1)
                done_rollout = truncated[0]  # rollout until truncated
                # update statistics (only count the envs that were not terminated)
                steps += self.n_env - term.sum().item()
                running_rewards += ((1 - term) * reward).sum().item()

                # determine success and termination
                if "success" in info:
                    s = ptu.from_numpy(info["success"]).float().view(self.n_env, 1)  # (n_env,1)
                    episode_success = torch.max(episode_success, s)  # (n_env,1) if success previously, keep it
                    if hasattr(self.config_env, "terminate_after_success"):
                        term = torch.max(term, episode_success)  # if success, set term to 1.0
                term = torch.max(term, ptu.FloatTensor(terminated).view(self.n_env, -1))  # (n_env, 1) if term previously, keep it
                if done_rollout and self.config_env.horizon == "finite":
                    term = ptu.ones_like(term)  # if finite horizon, set term to 1.0 at the end of episode

                if mode == "train":
                    # add data to policy buffer
                    obs_list.append(obs)  # (n_env, dim)
                    act_list.append(action)  # (n_env, dim)
                    rew_list.append(reward)  # (n_env, dim)
                    term_list.append(term)  # bool
                    next_obs_list.append(next_obs)  # (n_env, dim)
                    if self.config_rl.algo == "ppo":
                        logprob_list.append(logprob) # (n_env, dim)
                        value_list.append(value)
                else: # eval
                    if visualize and idx == 0:
                        frame = self.eval_env.render()[0]
                        frames.append(frame)

                # set: prev_obs<- obs, obs <- next_obs
                prev_obs = obs.clone()
                obs = next_obs.clone()

            if mode == "train":
                # add collected sequence to buffer
                act_buffer = torch.stack(act_list, dim=0)  # (L, n_env, dim)
                if not self.act_continuous:
                    act_buffer = torch.argmax(
                        act_buffer, dim=-1, keepdims=True
                    )  # (L, n_env, 1)
                obs_buffer = torch.stack(obs_list, dim=0)  # (L, n_env, dim)
                rewards_buffer = torch.stack(rew_list, dim=0)  # (L, n_env, 1)
                term_buffer = torch.stack(term_list, dim=0)  # (L, n_env, 1)
                obs_next_buffer = torch.stack(next_obs_list, dim=0)  # (L, n_env, dim)
                self.policy_storage.add_episode(
                    actions=act_buffer,
                    observations=obs_buffer,
                    next_observations=obs_next_buffer,
                    rewards=rewards_buffer,
                    terminals=term_buffer,
                    values=torch.stack(value_list, dim=0) if self.config_rl.algo == "ppo" else None,
                    logprobs=torch.stack(logprob_list, dim=0) if self.config_rl.algo == "ppo" else None
                )

                self._n_env_steps_total += steps
                self._n_episodes_total += self.n_env

            returns_per_episode[idx] = running_rewards / self.n_env
            success_rate[idx] = episode_success.mean().item()
            avg_steps[idx] = steps / self.n_env
        d_rollout = {"return": np.mean(returns_per_episode), "success_rate": np.mean(success_rate), "episode_len": np.mean(avg_steps)}
        self.agent.train()  # set it back to train
        if mode == "train":
            return d_rollout, self._n_env_steps_total - before_env_steps
        else: # eval
            return d_rollout, frames


    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.policy_storage.random_episodes(self.FLAGS.batch_size)

            # RL update
            rl_losses = self.agent.update(batch)

            for k, v in rl_losses.items():
                if not torch.is_tensor(v):
                    v = ptu.tensor(v)
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # statistics
        for k in rl_losses_agg:
            rl_losses_agg[k] = torch.stack(rl_losses_agg[k]).mean(dim=0)
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg
    
    def update_ppo(self):
        return self.agent.update(self.policy_storage)
