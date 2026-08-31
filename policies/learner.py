import time
import os
import math
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.nn import functional as F

from envs.alchemy import (
    ALCHEMY_ACTION_CATEGORY_CASH,
    ALCHEMY_ACTION_CATEGORY_NAMES,
    ALCHEMY_ACTION_CATEGORY_POTION,
    action_category_ids,
    get_symbolic_alchemy_layout,
    valid_action_mask_from_observation,
)

from .models import AGENT_CLASSES
from .rollout import EpisodeTrajectory, RolloutResult

from buffers.rollout_buffer import RolloutBuffer

from torchkit import pytorch_utils as ptu
from utils.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    load_training_checkpoint,
    save_verified,
)
import matplotlib.pyplot as plt
import wandb


class _AlchemyRolloutDiagnostics:
    def __init__(self, *, observe_used, add_trial_flag, context_dim,
                 structured_potions=False):
        self.observe_used = bool(observe_used)
        self.mask_kwargs = {
            "observe_used": self.observe_used,
            "add_trial_flag": bool(add_trial_flag),
            "context_dim": int(context_dim),
            "structured_potions": bool(structured_potions),
        }
        self.has_data = False
        self.category_counts = torch.zeros(
            len(ALCHEMY_ACTION_CATEGORY_NAMES),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.invalid_count = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.valid_count_sum = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.valid_count_min = torch.full(
            (),
            float("inf"),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.valid_count_max = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.cash_nonzero_reward_count = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.potion_nonzero_reward_count = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )
        self.total_decisions = torch.zeros(
            (),
            dtype=torch.float32,
            device=ptu.device,
        )

    def update(self, *, obs, action, reward, active_mask):
        active_indices = torch.nonzero(
            active_mask.view(-1) > 0.5,
            as_tuple=False,
        ).squeeze(-1)
        if active_indices.numel() == 0:
            return

        action_ids = torch.argmax(action, dim=-1).view(-1)
        valid_action_mask = valid_action_mask_from_observation(
            obs,
            **self.mask_kwargs,
        )
        if valid_action_mask.shape[-1] != action.shape[-1]:
            raise ValueError(
                f"Alchemy action mask width {valid_action_mask.shape[-1]} "
                f"does not match action width {action.shape[-1]}"
            )
        valid_action_mask = valid_action_mask.view(
            -1,
            valid_action_mask.shape[-1],
        )

        selected_action_ids = action_ids.index_select(
            0,
            active_indices,
        ).long()
        selected_rewards = reward.view(-1).index_select(0, active_indices)
        selected_valid_action_mask = valid_action_mask.index_select(
            0,
            active_indices,
        )
        valid_action_count = selected_valid_action_mask.sum(dim=-1).to(
            dtype=torch.float32,
        )
        selected_valid = selected_valid_action_mask.gather(
            -1,
            selected_action_ids.unsqueeze(-1),
        ).squeeze(-1)
        selected_categories = action_category_ids(
            selected_action_ids,
            observe_used=self.observe_used,
        ).to(dtype=torch.long)
        category_count = torch.bincount(
            selected_categories,
            minlength=len(ALCHEMY_ACTION_CATEGORY_NAMES),
        )
        nonzero_reward = selected_rewards != 0

        self.has_data = True
        self.category_counts += category_count.to(dtype=torch.float32)
        self.invalid_count += (~selected_valid).float().sum()
        self.valid_count_sum += valid_action_count.sum()
        self.valid_count_min = torch.minimum(
            self.valid_count_min,
            valid_action_count.min(),
        )
        self.valid_count_max = torch.maximum(
            self.valid_count_max,
            valid_action_count.max(),
        )
        self.cash_nonzero_reward_count += (
            (selected_categories == ALCHEMY_ACTION_CATEGORY_CASH)
            & nonzero_reward
        ).float().sum()
        self.potion_nonzero_reward_count += (
            (selected_categories == ALCHEMY_ACTION_CATEGORY_POTION)
            & nonzero_reward
        ).float().sum()
        self.total_decisions += valid_action_count.new_tensor(
            selected_action_ids.numel()
        )

    def finalize(self):
        if not self.has_data:
            return {}

        stats = torch.stack(
            (
                self.total_decisions,
                self.invalid_count,
                self.valid_count_sum,
                self.valid_count_min,
                self.valid_count_max,
                self.cash_nonzero_reward_count,
                self.potion_nonzero_reward_count,
                *self.category_counts,
            )
        ).cpu().numpy()
        total_decisions = float(stats[0])
        if total_decisions <= 0.0:
            return {}

        invalid_count = int(round(float(stats[1])))
        valid_count_sum = float(stats[2])
        valid_count_min = int(round(float(stats[3])))
        valid_count_max = int(round(float(stats[4])))
        cash_nonzero_reward_count = int(round(float(stats[5])))
        potion_nonzero_reward_count = int(round(float(stats[6])))
        category_counts = stats[7:]

        metrics = {
            "alchemy/invalid_action_count": invalid_count,
            "alchemy/invalid_action_rate": (
                invalid_count / total_decisions
            ),
            "alchemy/valid_action_count_mean": (
                valid_count_sum / total_decisions
            ),
            "alchemy/valid_action_count_min": valid_count_min,
            "alchemy/valid_action_count_max": valid_count_max,
            "alchemy/cash_nonzero_reward_count": (
                cash_nonzero_reward_count
            ),
            "alchemy/cash_nonzero_reward_rate": (
                cash_nonzero_reward_count / total_decisions
            ),
            "alchemy/potion_nonzero_reward_count": (
                potion_nonzero_reward_count
            ),
            "alchemy/potion_nonzero_reward_rate": (
                potion_nonzero_reward_count / total_decisions
            ),
        }
        for index, name in enumerate(ALCHEMY_ACTION_CATEGORY_NAMES):
            count = int(round(float(category_counts[index])))
            metrics[f"alchemy/action_{name}_count"] = count
            metrics[f"alchemy/action_{name}_rate"] = (
                count / total_decisions
            )
        return metrics


class Learner:
    def __init__(self, env, eval_env, FLAGS, config_rl, config_seq, config_env):
        self.train_env = env
        self.eval_env = eval_env
        self.FLAGS = FLAGS
        self.config_rl = config_rl
        self.config_seq = config_seq
        self.config_env = config_env
        self.skip_reset_transition = bool(
            config_seq.get("skip_reset_transition", False)
        )
        self.resume_dir = getattr(FLAGS, 'resume', '')

        self.init_env()

        self.init_agent()

        self.init_train()

    def init_env(
        self,
    ):
        # get action / observation dimensions
        action_space = self.train_env.get_attr("action_space")[0]
        observation_space = self.train_env.get_attr("observation_space")[0]
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            self.act_continuous = True
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n
            self.act_continuous = False
        else:
            raise ValueError(
                f"Unsupported action space type: {type(action_space).__name__}"
            )
        self.obs_dim = observation_space.shape[0]
        self.n_env = self.config_env.n_env
        print("obs space", observation_space)
        print("act space", action_space)
        _, info = self.train_env.reset()
        context = info.get("context", None)
        if context is not None:
            context_dim = context.shape[-1]
        else:
            context_dim = 0
        self.config_seq.seq_model.context_dim = context_dim
        print("obs_dim", self.obs_dim, "act_dim", self.act_dim, "context_dim", context_dim, "n_env", self.n_env)

        # Per-attempt adaptation curves: "attempts" = k-shot attempts
        # (KEpisodeWrapper) or an env's native trials (Alchemy num_trials); at
        # most one is >1. inner_max is the per-attempt length.
        self.k = int(getattr(self.FLAGS, "k", 1))
        max_episode_steps = self.train_env.get_attr("max_episode_steps")[0]
        self.n_attempts = max(self.k, int(self.config_env.get("num_trials", 1)))
        self.inner_max = max_episode_steps // self.n_attempts
        if max_episode_steps % self.n_attempts != 0:
            raise ValueError(
                "max_episode_steps must be divisible by the number of attempts"
            )
        self._alchemy_rollout_mask_kwargs = (
            self._infer_alchemy_rollout_mask_kwargs()
        )

    def _infer_alchemy_rollout_mask_kwargs(self):
        if (
            getattr(self.config_env, "env_type", None) != "alchemy"
            or self.act_continuous
        ):
            return None
        observe_used = bool(
            getattr(self.config_env, "observe_used", True)
        )
        add_trial_flag = bool(
            getattr(self.config_env, "add_trial_flag", False)
        )
        structured_potions = bool(
            getattr(self.config_env, "structured_potions", False)
        )
        layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
        symbolic_obs_dim = (
            layout.symbolic_obs_dim + int(add_trial_flag)
        )
        context_dim = self.obs_dim - symbolic_obs_dim
        if context_dim < 0:
            raise ValueError(
                "Alchemy observation width is smaller than the symbolic "
                f"layout: obs_dim={self.obs_dim}, "
                f"symbolic_obs_dim={symbolic_obs_dim}"
            )
        return {
            "observe_used": observe_used,
            "add_trial_flag": add_trial_flag,
            "context_dim": context_dim,
            "structured_potions": structured_potions,
        }

    def _create_rollout_diagnostics(self):
        if not hasattr(self, "_alchemy_rollout_mask_kwargs"):
            self._alchemy_rollout_mask_kwargs = (
                self._infer_alchemy_rollout_mask_kwargs()
            )
        if self._alchemy_rollout_mask_kwargs is None:
            return None
        return _AlchemyRolloutDiagnostics(
            **self._alchemy_rollout_mask_kwargs
        )

    def init_agent(
        self,
    ):
        try:
            agent_class = AGENT_CLASSES[self.config_rl.algo]
        except KeyError as exc:
            supported = ", ".join(sorted(AGENT_CLASSES))
            raise ValueError(
                f"Unknown RL algorithm {self.config_rl.algo!r}; "
                f"expected one of: {supported}"
            ) from exc

        self.agent = agent_class(
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            config_seq=self.config_seq,
            config_rl=self.config_rl,
            config_env=self.config_env,
            freeze_critic=self.FLAGS.freeze_critic,
        ).to(ptu.device)


    def init_train(
        self,
    ):
        self.msc_updates_per_rl = 0
        if self.agent.alternating_msc:
            msc_updates_per_rl = getattr(
                self.config_seq.seq_model, "msc_updates_per_rl", None
            )
            if (
                isinstance(msc_updates_per_rl, bool)
                or not isinstance(msc_updates_per_rl, int)
                or msc_updates_per_rl <= 0
            ):
                raise ValueError(
                    "config_seq.seq_model.msc_updates_per_rl must be a "
                    "positive integer when alternating MSC is enabled"
                )
            self.msc_updates_per_rl = msc_updates_per_rl

        num_episodes = int(self.config_rl.replay_buffer_num_episodes)
        obs_backend = getattr(self.config_env, "obs_backend", "ram")
        obs_dtype = getattr(self.config_env, "obs_dtype", "float32")
        memmap_dir = None
        if obs_backend == "memmap":
            # Use SCRATCH local SSD for fast disk I/O (avoid blob FUSE slowness + storage hiccups).
            # Buffer is ephemeral - lost on preempt-resume but training_checkpoint.pth still on blob.
            scratch_base = "/scratch/buffer_mmap" if os.path.isdir("/scratch") else None
            if scratch_base is not None:
                run_id = os.path.basename(self.FLAGS.log_dir.rstrip("/")) or "default_run"
                memmap_dir = os.path.join(scratch_base, run_id)
                print(f"[memmap] Using SCRATCH local: {memmap_dir}")
            else:
                memmap_dir = os.path.join(self.FLAGS.log_dir, "buffer_mmap")
                print(f"[memmap] Using blob path (no /scratch): {memmap_dir}")
        self.policy_storage = RolloutBuffer(
            observation_dim=self.obs_dim,
            action_dim=self.act_dim if self.act_continuous else None,
            max_episode_len=self.train_env.get_attr("max_episode_steps")[0],
            num_episodes=num_episodes,
            obs_backend=obs_backend,
            obs_dtype=obs_dtype,
            memmap_dir=memmap_dir,
            max_seq_len=getattr(self.FLAGS, "max_seq_len", -1),
            require_memory_masks=self.skip_reset_transition,
        )

        self.total_episodes = self.FLAGS.start_training + self.FLAGS.train_episodes

    def save_checkpoint(self):
        ckpt = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "agent_training_state": self.agent.training_state_dict(),
            "counters": {
                "env_steps": self._n_env_steps_total,
                "rl_updates": self._n_rl_update_steps_total,
                "msc_updates": self._n_msc_update_steps_total,
                "episodes": self._n_episodes_total,
            },
            "wandb_run_id": wandb.run.id,
            "wandb_run_name": wandb.run.name,
            "config": {
                "config_env": self.config_env.to_dict(),
                "config_rl": self.config_rl.to_dict(),
                "config_seq": self.config_seq.to_dict(),
            },
        }
        save_verified(ckpt, f"{self.FLAGS.log_dir}/training_checkpoint.pth")
        save_verified(
            self.policy_storage.state_dict(),
            f"{self.FLAGS.log_dir}/buffer_checkpoint.pth",
        )

    def load_checkpoint(self, resume_dir):
        ckpt = load_training_checkpoint(
            f"{resume_dir}/training_checkpoint.pth",
            map_location=ptu.device,
        )
        self.agent.load_training_state_dict(ckpt["agent_training_state"])
        counters = ckpt["counters"]
        self._n_env_steps_total = counters["env_steps"]
        self._n_rl_update_steps_total = counters["rl_updates"]
        self._n_msc_update_steps_total = counters.get("msc_updates", 0)
        self._n_episodes_total = counters["episodes"]

        self.policy_storage.load_state_dict(
            torch.load(f"{resume_dir}/buffer_checkpoint.pth", map_location="cpu", weights_only=False))
        print(
            f"Resumed: episodes={self._n_episodes_total}, "
            f"env_steps={self._n_env_steps_total}, "
            f"rl_updates={self._n_rl_update_steps_total}, "
            f"msc_updates={self._n_msc_update_steps_total}"
        )

    def _start_training(self):
        self._start_time = time.time()
        if self.resume_dir:
            self.load_checkpoint(self.resume_dir)
        else:
            self._n_env_steps_total = 0
            self._n_rl_update_steps_total = 0
            self._n_msc_update_steps_total = 0
            self._n_episodes_total = 0


    def train(self):
        self._start_training()

        if not self.resume_dir and self.FLAGS.start_training > 0:
            while self._n_episodes_total < self.FLAGS.start_training:
                self.collect_rollouts(num_rollouts=1, random_actions=True)

            self.update(
                int(self._n_env_steps_total * self.FLAGS.updates_per_step)
            )
        try:
            while self._n_episodes_total < self.total_episodes:
                d_rollout, env_steps = self.collect_rollouts(num_rollouts=1)
                d_update = self.update(
                    int(math.ceil(self.FLAGS.updates_per_step * env_steps))
                )
                if self._n_episodes_total % self.config_env.log_interval == 0:
                    self._log_training(d_rollout, d_update)
                if self._n_episodes_total % self.config_env.eval_interval == 0:
                    self._evaluate_and_checkpoint()
        finally:
            self.policy_storage.close()

    def _log_training(self, rollout_metrics, update_metrics):
        visualize = (
            self._n_episodes_total
            % (self.config_env.visualize_every * self.config_env.log_interval)
            == 0
        )
        log_data = self.process_and_log_train(
            {**rollout_metrics, **update_metrics},
            visualize=visualize,
        )
        log_data.update(self._compute_param_norms())
        log_data.update(
            {
                "info/env_steps": self._n_env_steps_total,
                "info/rl_update_steps": self._n_rl_update_steps_total,
                "info/msc_update_steps": self._n_msc_update_steps_total,
                "info/duration_minute": (time.time() - self._start_time) / 60,
            }
        )
        wandb.log(log_data, self._n_episodes_total)

    def _evaluate_and_checkpoint(self):
        visualize = (
            self._n_episodes_total
            % (self.config_env.visualize_every * self.config_env.eval_interval)
            == 0
            and self.config_env.visualize_env
        )
        rollout_metrics, frames = self.collect_rollouts(
            num_rollouts=self.config_env.eval_episodes // self.n_env,
            mode="eval",
            visualize=visualize,
            deterministic=True,
        )
        print(
            f"Total rollouts:{self._n_episodes_total}, "
            f"Return: {rollout_metrics['return']:.2f}, "
            f"Success rate: {rollout_metrics['success_rate']:.2f}, "
            f"Episode_len: {rollout_metrics['episode_len']:.2f}"
        )
        eval_log = {
            f"eval/{key}": value
            for key, value in rollout_metrics.items()
        }
        if frames is not None:
            eval_log["eval/visualization"] = wandb.Video(
                np.array(frames).transpose(0, 3, 1, 2),
                fps=30,
                format="gif",
            )
        wandb.log(eval_log, self._n_episodes_total)
        self.save_checkpoint()
    def process_and_log_train(self, d_train, visualize=False):
        """
        Processes and log training data.

        If visualize is True:
            - For (T,) data, generate matplotlib Figure objects for
              metric vs. time (t) plots.
        Scalar metrics are retained as-is (not processed).
        """
        d_log = {}
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

                    d_log["visualizations/" + key] = wandb.Image(fig)

                    plt.close(fig)
            else:
                # scalar metrics: convert GPU tensors to Python scalars for logging
                if isinstance(value, torch.Tensor):
                    value = value.item()
                d_log["train/" + key] = value
        return d_log

    @torch.no_grad()
    def _compute_param_norms(self):
        """Per-parameter L2 weight norms (+ a global norm) for logging.

        Called once per `log_interval` (not per update), so the cost is amortized
        over the many gradient steps between logs. Target networks and frozen EMA
        parameters are skipped. All norms are computed on-device and moved to CPU
        in a single transfer to avoid per-parameter GPU-CPU syncs.
        """
        names, norms = [], []
        for name, p in self.agent.named_parameters():
            if "_target" in name or not p.requires_grad:
                continue
            names.append(name)
            norms.append(p.detach().norm())
        if not norms:
            return {}
        stacked = torch.stack(norms).cpu()  # single GPU->CPU sync for all params
        d = {f"param_norm/{n}": stacked[i].item() for i, n in enumerate(names)}
        d["param_norm/_global"] = stacked.norm().item()
        return d



    @torch.no_grad()
    def collect_rollouts(
        self,
        num_rollouts,
        random_actions=False,
        mode="train",
        deterministic=False,
        visualize=False,
    ):
        if num_rollouts < 1:
            raise ValueError("num_rollouts must be at least 1")
        if mode not in {"train", "eval"}:
            raise ValueError(f"Unknown rollout mode {mode!r}")
        if visualize or deterministic:
            if mode != "eval":
                raise ValueError(
                    "Visualization and deterministic actions are only supported "
                    "for evaluation rollouts"
                )

        self.agent.eval()
        before_env_steps = self._n_env_steps_total
        returns_per_episode = np.zeros(num_rollouts)
        success_rate = np.zeros(num_rollouts)
        avg_steps = np.zeros(num_rollouts)
        if self.n_attempts > 1:
            returns_attempt = np.zeros((num_rollouts, self.n_attempts))
            success_attempt = np.zeros((num_rollouts, self.n_attempts))
        frames = None
        rewards = None
        rollout_diagnostics = self._create_rollout_diagnostics()
        current_env = self.train_env if mode == "train" else self.eval_env
        try:
            for idx in range(num_rollouts):
                result = self._collect_single_rollout(
                    current_env=current_env,
                    mode=mode,
                    random_actions=random_actions,
                    deterministic=deterministic,
                    capture_frames=visualize and idx == 0,
                    rollout_diagnostics=rollout_diagnostics,
                )
                returns_per_episode[idx] = result.episode_return
                success_rate[idx] = result.success_rate
                avg_steps[idx] = result.average_steps
                rewards = result.rewards
                if result.frames is not None:
                    frames = result.frames
                if self.n_attempts > 1:
                    returns_attempt[idx] = result.attempt_returns
                    success_attempt[idx] = result.attempt_success
        finally:
            self.agent.train()

        metrics = {
            "return": np.mean(returns_per_episode),
            "success_rate": np.mean(success_rate),
            "episode_len": np.mean(avg_steps),
        }
        if self.n_attempts > 1:
            for attempt in range(self.n_attempts):
                metrics[f"return_attempt_{attempt}"] = returns_attempt[:, attempt].mean()
                metrics[f"success_attempt_{attempt}"] = success_attempt[:, attempt].mean()
            metrics.update(self._adaptation_metrics(returns_attempt))
        if rollout_diagnostics is not None:
            metrics.update(rollout_diagnostics.finalize())

        if mode == "train":
            metrics["reward"] = rewards.squeeze(-1).mean(-1)
            return metrics, self._n_env_steps_total - before_env_steps
        return metrics, frames

    def _adaptation_metrics(self, returns_attempt):
        """Within-episode learning: mean(late attempts) - mean(early attempts).

        The headline number for meta-RL envs like Alchemy, where the raw return
        is dominated by reward the agent gets without learning anything. A
        scripted policy scores ~0 here by construction, so anything positive is
        the memory adapting to the episode's hidden task. Uses the first/last
        third of the attempts (at least one each).
        """
        window = max(1, self.n_attempts // 3)
        early = returns_attempt[:, :window].mean()
        late = returns_attempt[:, -window:].mean()
        return {
            "adaptation": late - early,
            "adaptation_window": window,
        }

    def _collect_single_rollout(
        self,
        *,
        current_env,
        mode,
        random_actions,
        deterministic,
        capture_frames,
        rollout_diagnostics,
    ) -> RolloutResult:
        steps = 0
        running_rewards = 0.0
        obs = ptu.from_numpy(current_env.reset()[0])
        episode_success = ptu.zeros((self.n_env, 1))
        attempt_returns = None
        attempt_success = None
        if self.n_attempts > 1:
            attempt_returns = ptu.zeros((self.n_env, self.n_attempts))
            attempt_success = ptu.zeros((self.n_env, self.n_attempts))

        prev_obs, action, reward, term = self.get_initial_dummies(
            current_env,
            obs,
        )
        trajectory = None
        if mode == "train":
            trajectory = EpisodeTrajectory.start(
                prev_obs=prev_obs,
                action=action,
                reward=reward,
                obs=obs,
                terminal=term,
                memory_mask=ptu.ones((self.n_env, 1)),
            )

        frames = [] if capture_frames else None
        internal_state = None
        initial = True
        pending_memory_skip = False
        timestep = 0
        done_rollout = False

        while not done_rollout:
            if random_actions:
                action = self._sample_random_action(
                    current_env,
                    raw_obs=obs,
                )
            else:
                action, internal_state = self.act(
                    internal_state,
                    action,
                    reward,
                    prev_obs,
                    obs,
                    deterministic,
                    initial,
                    timestep,
                    pending_memory_skip,
                )
            initial = False

            env_action = self._to_env_action(action, current_env)
            next_obs, reward, terminated, truncated, info = current_env.step(
                env_action
            )
            next_obs = ptu.from_numpy(next_obs).view(self.n_env, -1)
            reward = ptu.from_numpy(reward).view(self.n_env, -1)
            done_rollout = bool(truncated[0])
            soft_reset = self._soft_reset_mask(info)

            steps += self.n_env - term.sum().item()
            running_rewards += ((1 - term) * reward).sum().item()
            if rollout_diagnostics is not None:
                rollout_diagnostics.update(
                    obs=obs,
                    action=action,
                    reward=reward,
                    active_mask=term < 0.5,
                )

            attempt_index = None
            if self.n_attempts > 1:
                attempt_index = timestep // self.inner_max
                attempt_returns[:, attempt_index] += reward.view(-1)

            if "success" in info:
                success = ptu.from_numpy(info["success"]).float().view(
                    self.n_env,
                    1,
                )
                episode_success = torch.maximum(episode_success, success)
                if self.n_attempts > 1:
                    attempt_success[:, attempt_index] = torch.maximum(
                        attempt_success[:, attempt_index],
                        success.view(-1),
                    )
                if self.config_env.get("terminate_after_success", True):
                    term = torch.maximum(term, episode_success)

            term = torch.maximum(
                term,
                ptu.from_numpy(terminated).view(self.n_env, -1),
            )
            if done_rollout and self.config_env.horizon == "finite":
                term = ptu.ones_like(term)

            if trajectory is not None:
                trajectory.append(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    terminal=term,
                    memory_mask=ptu.from_numpy(
                        (~soft_reset).astype(np.float32)
                    ).view(self.n_env, 1),
                )
            if frames is not None:
                frames.append(current_env.render()[0])

            prev_obs = obs
            obs = next_obs
            timestep += 1
            pending_memory_skip = (
                self.skip_reset_transition and bool(soft_reset[0])
            )

        rewards = None
        if trajectory is not None:
            rewards = trajectory.commit(
                self.policy_storage,
                continuous_actions=self.act_continuous,
            )
            self._n_env_steps_total += steps
            self._n_episodes_total += self.n_env

        return RolloutResult(
            episode_return=running_rewards / self.n_env,
            success_rate=episode_success.mean().item(),
            average_steps=steps / self.n_env,
            rewards=rewards,
            frames=frames,
            attempt_returns=(
                ptu.get_numpy(attempt_returns.mean(dim=0))
                if attempt_returns is not None
                else None
            ),
            attempt_success=(
                ptu.get_numpy(attempt_success.mean(dim=0))
                if attempt_success is not None
                else None
            ),
        )

    def _soft_reset_mask(self, info):
        soft_reset = np.asarray(
            info.get("soft_reset", np.zeros(self.n_env, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if soft_reset.size != self.n_env:
            raise ValueError(
                f"soft_reset has {soft_reset.size} entries; expected {self.n_env}"
            )
        if self.skip_reset_transition and not np.all(
            soft_reset == soft_reset[0]
        ):
            raise ValueError(
                "skip_reset_transition requires synchronized vector-env boundaries"
            )
        return soft_reset

    def _sample_random_action(self, current_env, raw_obs=None):
        if (
            not self.act_continuous
            and getattr(
                self.agent,
                "mask_alchemy_invalid_actions",
                False,
            )
        ):
            return self.agent.sample_random_action(raw_obs=raw_obs)
        action = ptu.from_numpy(current_env.action_space.sample()).reshape(
            self.n_env,
            -1,
        )
        if not self.act_continuous:
            action = F.one_hot(
                action.squeeze(-1).long(),
                num_classes=self.act_dim,
            ).float()
        return action

    def _to_env_action(self, action, current_env):
        env_action = ptu.get_numpy(action)
        if not self.act_continuous:
            env_action = np.argmax(env_action, axis=-1)
        if not current_env.action_space.contains(env_action):
            raise ValueError("Agent produced an action outside the environment space")
        return env_action

    def act(
        self, internal_state, action, reward, prev_obs, obs, deterministic,
        initial, timestep=0, skip_memory_update=False,
    ):
        action, internal_state = self.agent.act(
            prev_internal_state=internal_state,
            prev_action=action,
            prev_reward=reward,
            prev_obs=prev_obs,
            obs=obs,
            deterministic=deterministic,
            initial=initial,
            timestep=timestep,
            skip_memory_update=skip_memory_update,
        )

        return action, internal_state

    def get_initial_dummies(self, current_env, obs):
        prev_obs = obs.clone()
        action = self._sample_random_action(current_env, raw_obs=obs)
        reward = ptu.zeros((self.n_env, 1))
        term = ptu.zeros((self.n_env, 1))
        return prev_obs, action, reward, term


    def update(self, num_updates):
        if num_updates < 0:
            raise ValueError("num_updates must be non-negative")
        if num_updates == 0:
            return {}
        rl_losses_agg = {}
        for _ in range(num_updates):
            for _ in range(self.msc_updates_per_rl):
                batch = self.policy_storage.random_episodes(
                    self.FLAGS.batch_size
                )
                msc_losses = self.agent.update_msc(batch)
                self._n_msc_update_steps_total += 1

                for key, value in msc_losses.items():
                    if not torch.is_tensor(value):
                        value = ptu.tensor(value)
                    rl_losses_agg.setdefault(key, []).append(value)

            batch = self.policy_storage.random_episodes(self.FLAGS.batch_size)
            rl_losses = self.agent.update(batch)

            for key, value in rl_losses.items():
                if not torch.is_tensor(value):
                    value = ptu.tensor(value)
                rl_losses_agg.setdefault(key, []).append(value)

        rl_losses_agg = {
            key: torch.stack(values).mean(dim=0)
            for key, values in rl_losses_agg.items()
        }
        self._n_rl_update_steps_total += num_updates

        return rl_losses_agg
