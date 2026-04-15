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

# logging settings
flags.DEFINE_string('run_name', 'test', 'A unique name for this run.')
flags.DEFINE_string("save_dir", "logs", "logging dir.")
flags.DEFINE_string("resume", "", "Path to log_dir to resume training from.")


def main(argv):
        
    seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    is_oracle = config_seq.seq_model.get("is_oracle", False)
    env = AsyncVectorEnv([lambda i=i: make_env(env_name, seed + i, mode="train", is_oracle=is_oracle,
                                               max_episode_steps=config_env.get("max_episode_steps")) for i in range(config_env.n_env)],
                         autoreset_mode= gym.vector.AutoresetMode.DISABLED) # codebase is designed for non-autoreset environments
    config_env.visualize_env = config_env.get("visualize_env", False)
    eval_env = AsyncVectorEnv([lambda i=i: make_env(env_name, seed + config_env.n_env + 42 + i, mode = "train", is_oracle=is_oracle,
                                                    visualize = config_env.visualize_env and i == 0,
                                                    max_episode_steps=config_env.get("max_episode_steps")) for i in range(config_env.n_env)],
                             autoreset_mode= gym.vector.AutoresetMode.DISABLED)


    system.reproduce(seed)
    set_gpu_mode(torch.cuda.is_available(), FLAGS.device)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high') # Use TF32 for faster matmul

    config_seq = config_seq.update_fn(config_seq, env.get_attr("max_episode_steps")[0])
    max_training_steps = int(FLAGS.train_episodes * env.get_attr("max_episode_steps")[0])
    config_rl = config_rl.update_fn(config_rl, env.get_attr("max_episode_steps")[0], max_training_steps)
    validate_flags(FLAGS)

    configs = {"config_env": FLAGS.config_env.to_dict(), "config_rl": FLAGS.config_rl.to_dict(), "config_seq": FLAGS.config_seq.to_dict()}

    if FLAGS.resume:
        log_dir = FLAGS.resume
        assert os.path.isdir(log_dir), f"Resume dir not found: {log_dir}"
        ckpt = torch.load(f"{log_dir}/training_checkpoint.pth", map_location="cpu", weights_only=False)
        validate_resume_config(ckpt["config"], configs)
        FLAGS.log_dir = log_dir
        wandb.init(project=env_name, id=ckpt["wandb_run_id"], resume="must", dir=log_dir, config=configs)
        del ckpt
    else:
        run_name = f"{config_env.env_type}/{config_env.env_name}/{FLAGS.run_name}_{system.now_str()}"
        log_dir = os.path.join(FLAGS.save_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        FLAGS.log_dir = log_dir
        wandb.init(project=env_name, name=run_name, dir=log_dir, config=configs)
    
    # start training
    learner = Learner(env, eval_env, FLAGS, config_rl, config_seq, config_env)
    learner.train()

def validate_flags(FLAGS):
    assert FLAGS.config_env.log_interval % FLAGS.config_env.n_env == 0 and FLAGS.config_env.eval_interval % FLAGS.config_env.n_env == 0, \
        "log_interval and eval_interval should be divisible by n_env."
    assert FLAGS.config_env.eval_episodes % FLAGS.config_env.n_env == 0, \
        "eval_episodes should be divisible by n_env."

def validate_resume_config(saved_config, current_config):
    """Validate that resumed configs match the checkpoint (except derived fields)."""
    skip = {("config_rl", "schedule_steps"), ("config_rl", "replay_buffer_num_episodes")}
    for sec in ["config_env", "config_rl", "config_seq"]:
        for k, cv in current_config[sec].items():
            if (sec, k) in skip or k not in saved_config[sec]:
                continue
            sv = saved_config[sec][k]
            if isinstance(cv, dict) and isinstance(sv, dict):
                for sk in cv:
                    if sk in sv and cv[sk] != sv[sk]:
                        raise ValueError(f"Config mismatch on resume: {sec}.{k}.{sk}")
            elif sv != cv:
                raise ValueError(f"Config mismatch on resume: {sec}.{k}")

if __name__ == "__main__":
    app.run(main)
