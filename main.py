import os
import torch
from absl import app, flags
from ml_collections import config_flags
from utils import system
import wandb

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner
from envs.make_env import make_env
from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
from utils.checkpointing import load_training_checkpoint
from utils.experiment import (
    finalize_training_configs,
    validate_resume_config,
    validate_run_settings,
)

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config_env",
    None,
    "File path to the environment configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_rl",
    None,
    "File path to the RL algorithm configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "config_seq",
    None,
    "File path to the seq model configuration.",
    lock_config=False,
)

flags.mark_flags_as_required(["config_rl", "config_env", "config_seq"])

# shared encoder settings
flags.DEFINE_boolean(
    "freeze_critic", True, "freeze critic params in actor loss"
)

# training settings
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("device", 0, "GPU device to use")
flags.DEFINE_integer("batch_size", 64, "Mini batch size.")
flags.DEFINE_integer("train_episodes", 1000, "Number of episodes during training.")
flags.DEFINE_float("updates_per_step", 0.1, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")
flags.DEFINE_integer(
    "max_seq_len", -1,
    "Number of real transitions used per training update. -1 (default) trains "
    "on all T transitions (T+1 replay rows including context). When shorter, "
    "config_seq.seq_model.truncated_sampling selects a sorted random 'subset' "
    "or contiguous 'window'. MATE defaults to subset; other models default to "
    "window. Subset sampling requires obs_shortcut=True. Inference always uses "
    "the full history.",
)
flags.DEFINE_integer(
    "k", 1,
    "k-shot: number of same-task attempts concatenated into one meta-episode "
    "(RL^2). 1 (default) disables it. Task/context is held fixed across attempts "
    "and a soft-reset flag is appended to the observation.",
)

# logging settings
flags.DEFINE_string('run_name', 'test', 'A unique name for this run.')
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("resume", "", "Path to log_dir to resume training from.")
flags.DEFINE_string("timestamp", "", "Override auto-generated timestamp (e.g., AMLT_EXPERIMENT_NAME for resume consistency). If empty, uses current time.")


def main(argv):
    del argv
    config_env = FLAGS.config_env
    config_env, env_name = config_env.create_fn(config_env)
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq
    is_oracle = config_seq.seq_model.get("is_oracle", False)

    if FLAGS.k > 1 and config_env.get("terminate_after_success", True):
        config_env.terminate_after_success = False
        print(f"[k-shot] k={FLAGS.k}: forcing config_env.terminate_after_success=False.")

    config_env.visualize_env = config_env.get("visualize_env", False)
    env, eval_env = create_vector_envs(
        env_name=env_name,
        config_env=config_env,
        seed=FLAGS.seed,
        is_oracle=is_oracle,
        k=FLAGS.k,
    )

    try:
        configure_runtime(FLAGS.seed, FLAGS.device)
        max_episode_steps = env.get_attr("max_episode_steps")[0]
        config_rl, config_seq = finalize_training_configs(
            config_rl,
            config_seq,
            max_episode_steps=max_episode_steps,
            train_episodes=FLAGS.train_episodes,
        )
        validate_run_settings(
            config_env,
            max_seq_len=FLAGS.max_seq_len,
            max_episode_steps=max_episode_steps,
        )
        initialize_run(
            env_name=env_name,
            config_env=config_env,
            config_rl=config_rl,
            config_seq=config_seq,
        )
        log_windowing(FLAGS.max_seq_len, max_episode_steps, config_seq.seq_model.truncated_sampling)

        learner = Learner(
            env,
            eval_env,
            FLAGS,
            config_rl,
            config_seq,
            config_env,
        )
        learner.train()
    finally:
        env.close()
        eval_env.close()
        if wandb.run is not None:
            wandb.finish()


def create_vector_envs(*, env_name, config_env, seed, is_oracle, k):
    train_seeds = [seed + index for index in range(config_env.n_env)]
    eval_seeds = [
        seed + config_env.n_env + 42 + index
        for index in range(config_env.n_env)
    ]
    common = {
        "env_name": env_name,
        "is_oracle": is_oracle,
        "k": k,
        "max_episode_steps": config_env.get("max_episode_steps"),
    }
    train_env = _create_vector_env(
        seeds=train_seeds,
        visualize_first=False,
        **common,
    )
    eval_env = _create_vector_env(
        seeds=eval_seeds,
        visualize_first=config_env.visualize_env,
        **common,
    )
    return train_env, eval_env


def _create_vector_env(
    *,
    env_name,
    seeds,
    is_oracle,
    k,
    max_episode_steps,
    visualize_first,
):
    env_fns = [
        (
            lambda env_seed=env_seed, visualize=(visualize_first and index == 0):
            make_env(
                env_name,
                env_seed,
                mode="train",
                is_oracle=is_oracle,
                k=k,
                visualize=visualize,
                max_episode_steps=max_episode_steps,
            )
        )
        for index, env_seed in enumerate(seeds)
    ]
    return AsyncVectorEnv(
        env_fns,
        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
    )


def configure_runtime(seed, device):
    system.reproduce(seed)
    use_cuda = torch.cuda.is_available()
    set_gpu_mode(use_cuda, device)
    if use_cuda:
        torch.set_float32_matmul_precision("high")


def log_windowing(max_seq_len, max_episode_steps, truncated_sampling):
    if 0 < max_seq_len < max_episode_steps:
        if truncated_sampling == "subset":
            mode = "sorted random transition subset"
        else:
            mode = "contiguous BPTT window"
        print(
            f"[max_seq_len] training {mode}={max_seq_len} "
            f"(episode length={max_episode_steps}); inference uses full history."
        )


def initialize_run(*, env_name, config_env, config_rl, config_seq):
    configs = {
        "config_env": config_env.to_dict(),
        "config_rl": config_rl.to_dict(),
        "config_seq": config_seq.to_dict(),
    }
    if FLAGS.resume:
        log_dir = FLAGS.resume
        if not os.path.isdir(log_dir):
            raise ValueError(f"Resume dir not found: {log_dir}")
        checkpoint = load_training_checkpoint(
            f"{log_dir}/training_checkpoint.pth",
            map_location="cpu",
        )
        validate_resume_config(checkpoint["config"], configs)
        FLAGS.log_dir = log_dir
        wandb.init(
            entity="mate_research",
            project=env_name,
            id=checkpoint["wandb_run_id"],
            name=checkpoint["wandb_run_name"],
            resume="must",
            dir=log_dir,
            config=configs,
        )
        return

    timestamp = FLAGS.timestamp or system.now_str()
    log_subpath = (
        f"{config_env.env_type}/{config_env.env_name}/"
        f"{FLAGS.run_name}_{timestamp}"
    )
    log_dir = os.path.join(FLAGS.save_dir, log_subpath)
    os.makedirs(log_dir, exist_ok=True)
    FLAGS.log_dir = log_dir
    wandb.init(
        entity="mate_research",
        project=env_name,
        name=FLAGS.run_name,
        dir=log_dir,
        config=configs,
    )

if __name__ == "__main__":
    app.run(main)
