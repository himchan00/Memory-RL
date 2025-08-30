import os
import time

import math
import numpy as np
import torch
from torch.nn import functional as F

from .models import AGENT_CLASSES

# RNN policy on image/vector-based task
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer
# For PPO
from buffers.rollout_buffer import RolloutBuffer

from utils import helpers as utl
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
        assert len(self.eval_env.observation_space.shape) == 1  # flatten
        if self.eval_env.action_space.__class__.__name__ == "Box":
            # continuous action space
            self.act_dim = self.eval_env.action_space.shape[0]
            self.act_continuous = True
        else:
            assert self.eval_env.action_space.__class__.__name__ == "Discrete"
            self.act_dim = self.eval_env.action_space.n
            self.act_continuous = False
        self.obs_dim = self.eval_env.observation_space.shape[0]
        self.n_env = self.config_env.n_env
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
            self.policy_storage = RolloutBuffer(observation_dim=self.obs_dim,
                                            action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                                            max_episode_len=self.eval_env.max_episode_steps,
                                            n_steps=self.FLAGS.batch_size)
        else:
            self.policy_storage = RAMEfficient_SeqReplayBuffer(
                max_replay_buffer_size=max(
                    int(self.config_rl.replay_buffer_size),
                    int(
                        self.config_rl.replay_buffer_num_episodes
                        * self.eval_env.max_episode_steps
                    ),
                ),
                observation_dim=self.obs_dim,
                action_dim=self.act_dim if self.act_continuous else 1,  # save memory
                sampled_seq_len=self.config_seq.sampled_seq_len,
                observation_type=self.eval_env.observation_space.dtype,
            )

        total_rollouts = self.FLAGS.start_training + self.FLAGS.train_episodes
        self.total_rollouts = total_rollouts

    def _start_training(self):
        self._n_env_steps_total = 0
        self._n_rl_update_steps_total = 0
        self._n_rollouts_total = 0
        self._start_time = time.time()


    def train(self):
        """
        training loop
        """

        self._start_training()

        if self.FLAGS.start_training > 0: # Set to 0 for PPO
            while self._n_rollouts_total < self.FLAGS.start_training:
                self.collect_rollouts(num_rollouts=1, random_actions=True)

            self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )

        while self._n_rollouts_total < self.total_rollouts:
            if self.config_rl.algo == "ppo":
                env_steps, d_rollout = self.collect_rollouts(num_rollouts=self.FLAGS.batch_size)
                d_update = self.update_ppo()
            else:
                env_steps, d_rollout = self.collect_rollouts(num_rollouts=1)
                d_update = self.update(
                    int(math.ceil(self.FLAGS.updates_per_step * env_steps))
                )  # NOTE: ceil to make sure at least 1 step


            if self._n_rollouts_total % self.config_env.log_interval == 0:
                # logging
                d_train = {**d_rollout, **d_update}
                visualize = self._n_rollouts_total % (self.config_env.visualize_every * self.config_env.log_interval) == 0
                d_train = self.process_and_log_train(d_train, visualize=visualize)
                d_info = {"info/env_steps": self._n_env_steps_total, "info/rl_update_steps": self._n_rl_update_steps_total, \
                            "info/duration_minute": (time.time() - self._start_time)/60}
                wandb.log(d_info, self._n_rollouts_total)

            # evaluate and log
            if self._n_rollouts_total % self.config_env.eval_interval == 0:
                visualize = self._n_rollouts_total % (self.config_env.visualize_every * self.config_env.eval_interval) == 0 and self.config_env.visualize_env
                returns_eval, success_rate_eval, total_steps_eval, frames = self.evaluate(visualization=visualize)
                avg_return, avg_success_rate, avg_episode_len = np.mean(returns_eval), np.mean(success_rate_eval), np.mean(total_steps_eval)
                d_eval = {"eval/return": avg_return, "eval/success_rate": avg_success_rate, "eval/episode_len": avg_episode_len}
                print(f"Total rollouts:{self._n_rollouts_total}, Return: {avg_return:.2f}, Success rate: {avg_success_rate:.2f}, Episode_len: {avg_episode_len:.2f}")
                wandb.log(d_eval, self._n_rollouts_total)
                if frames is not None:
                    wandb.log({"eval/visualization": wandb.Video(np.array(frames).transpose(0,3,1,2), fps=30, format="gif")}, self._n_rollouts_total)


    def process_and_log_train(self, d_train, visualize=False):
        """
        Processes and log training data.

        If visualize is True:
            - For (T,) data, generate matplotlib Figure objects for
              metric vs. time (t) plots.
        Scalar metrics are retained as-is (not processed).
        """

        for key, value in d_train.items():
            if visualize and isinstance(value, torch.Tensor) and value.ndim == 1:
                value = value.cpu().numpy()
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(value)
                ax.set_title(f"{key} vs t", fontsize = 20)
                ax.set_xlabel("t", fontsize = 16)
                ax.set_ylabel(f"{key}", fontsize = 16)
                ax.tick_params(axis='both', which='major', labelsize=14)
                plt.tight_layout()

                wandb.log({"visualizations/" + key : wandb.Image(fig)}, self._n_rollouts_total)

                plt.close(fig) 
            else:
                # scalar metrics are retained
                wandb.log({"train/" + key : value}, self._n_rollouts_total)
        return None



    @torch.no_grad()
    def collect_rollouts(self, num_rollouts, random_actions=False):
        """collect num_rollouts of trajectories in task and save into policy buffer
        :param random_actions: whether to use policy to sample actions, or randomly sample action space
        """
        self.agent.eval()  # set to eval mode for deterministic dropout
        before_env_steps = self._n_env_steps_total
        returns = 0

        for idx in range(num_rollouts):
            steps = 0

            obs = ptu.from_numpy(self.train_env.reset()[0])  # reset
            obs = obs.reshape(-1, obs.shape[-1]) # (n_env, obs_dim)
            done_rollout = False

            obs_list, act_list, rew_list, next_obs_list, term_list = (
                [],
                [],
                [],
                [],
                [],
            )
            if self.config_rl.algo == "ppo":
                logprob_list, value_list = ([], [])

            if not random_actions:
                # Dummy variables, not used
                prev_obs, action, reward, internal_state = self.agent.get_initial_info(batch_size = self.n_env)
                initial=True

            while not done_rollout:
                if random_actions:
                    action = ptu.FloatTensor([self.train_env.action_space.sample()]).reshape(self.n_env, -1)  # (B, A) for continuous action, (B, 1) for discrete action
                    if not self.act_continuous:
                        action = F.one_hot(
                            action.squeeze(-1).long(), num_classes=self.act_dim
                        ).float()  # (B, A)

                else:

                    if self.config_rl.algo == "ppo":
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
                            deterministic=False,
                            initial=initial,
                        )
                    initial=False

                # observe reward and next obs (B, dim)
                np_action = ptu.get_numpy(action)
                if not self.act_continuous:
                    np_action = np.argmax(np_action, axis=-1)  # one-hot to int
                if not self.train_env.action_space.contains(np_action):
                    raise ValueError("Invalid action!")
                next_obs, reward, terminated, truncated, info = self.train_env.step(np_action)
                next_obs = ptu.from_numpy(next_obs).view(self.n_env, -1)
                reward = ptu.FloatTensor(reward).view(self.n_env, -1)  # (B, 1)
                term = np.array(terminated, dtype=int)
                done_rollout = truncated[0] # rollout until truncated
                # update statistics
                steps += self.n_env

                # add data to policy buffer
                obs_list.append(obs)  # (n_env, dim)
                act_list.append(action)  # (n_env, dim)
                rew_list.append(reward)  # (n_env, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (n_env, dim)
                if self.config_rl.algo == "ppo":
                    logprob_list.append(logprob) # (n_env, dim)
                    value_list.append(value)

                # set: prev_obs<- obs, obs <- next_obs
                prev_obs = obs.clone()
                obs = next_obs.clone()

            # add collected sequence to buffer
            act_buffer = torch.cat(act_list, dim=0)  # (L, dim)
            if not self.act_continuous:
                act_buffer = torch.argmax(
                    act_buffer, dim=-1, keepdims=True
                )  # (L, 1)

            if self.config_rl.algo == "ppo":
                self.policy_storage.add_episode(
                    actions = act_buffer,
                    observations = torch.cat(obs_list, dim = 0),
                    next_observations = torch.cat(next_obs_list, dim = 0), 
                    logprobs = torch.cat(logprob_list, dim = 0), 
                    rewards = torch.cat(rew_list, dim = 0), 
                    values = torch.cat(value_list, dim = 0),
                    terminals = ptu.tensor(term_list).reshape(-1, 1)
                    )
            else:
                self.policy_storage.add_episode(
                    observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                    actions=ptu.get_numpy(act_buffer),  # (L, dim)
                    rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                    terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                    next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)),  # (L, dim)
                    )

            returns += torch.cat(rew_list, dim=0).sum() / self.n_env
            self._n_env_steps_total += steps
            self._n_rollouts_total += self.n_env

        avg_return = returns / num_rollouts
        avg_episode_len = (self._n_env_steps_total - before_env_steps) / num_rollouts / self.n_env
        d_rollout = {"return": avg_return, "episode_len": avg_episode_len}
        self.agent.train()  # set it back to train
        return self._n_env_steps_total - before_env_steps, d_rollout

    def sample_rl_batch(self, batch_size):
        batch = self.policy_storage.random_episodes(batch_size)
        return ptu.np_to_pytorch_batch(batch)

    def update(self, num_updates):
        rl_losses_agg = {}
        for update in range(num_updates):
            # sample random RL batch: in transitions
            batch = self.sample_rl_batch(self.FLAGS.batch_size)

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


    @torch.no_grad()
    def evaluate(self, deterministic=True, visualization = False):
        self.agent.eval()  # set to eval mode for deterministic dropout

        returns_per_episode = np.zeros(self.config_env.eval_episodes)
        success_rate = np.zeros(self.config_env.eval_episodes)
        total_steps = np.zeros(self.config_env.eval_episodes)

        for task_idx in range(self.config_env.eval_episodes):
            step = 0
            running_reward = 0.0
            done_rollout = False
            if visualization and task_idx == 0: # Visualization only for the first episode
                frames = []

            obs = ptu.from_numpy(self.eval_env.reset()[0])  # reset
            obs = obs.reshape(1, obs.shape[-1])

            # Dummy variables, not used
            prev_obs, action, reward, internal_state = self.agent.get_initial_info(batch_size = 1)
            initial=True

            while not done_rollout:
                action, internal_state = self.agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    prev_reward=reward,
                    prev_obs=prev_obs,
                    obs=obs,
                    deterministic=deterministic,
                    initial=initial
                )
                initial=False


                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(
                    self.eval_env, action.squeeze(dim=0)
                )
                if visualization and task_idx == 0:
                    frame = self.eval_env.render()
                    frames.append(frame)
                # add raw reward
                running_reward += reward.item()
                step += 1
                done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

                # set: prev_obs<- obs, obs <- next_obs
                prev_obs = obs.clone()
                obs = next_obs.clone()

            returns_per_episode[task_idx] = running_reward
            total_steps[task_idx] = step
            if "success" in info and info["success"] == True:
                success_rate[task_idx] = 1.0

        self.agent.train()  # set it back to train
        return returns_per_episode, success_rate, total_steps, frames if visualization else None

