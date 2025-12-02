import os
import torch
from absl import app, flags
from ml_collections import config_flags
import pickle
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
flags.DEFINE_float("updates_per_step", 0.25, "Gradient updates per step.")
flags.DEFINE_integer("start_training", 10, "Number of episodes to start training.")

# logging settings
flags.DEFINE_string('run_name', 'test', 'A unique name for this run.')
flags.DEFINE_string("save_dir", "logs", "logging dir.")


def main(argv):
        
    seed = FLAGS.seed

    config_env = FLAGS.config_env
    config_rl = FLAGS.config_rl
    config_seq = FLAGS.config_seq

    config_env, env_name = config_env.create_fn(config_env)
    env = AsyncVectorEnv([lambda i=i: make_env(env_name, seed + i, mode="train") for i in range(config_env.n_env)], 
                         autoreset_mode= gym.vector.AutoresetMode.DISABLED) # codebase is designed for non-autoreset environments
    config_env.visualize_env = config_env.get("visualize_env", False)
    eval_env = AsyncVectorEnv([lambda i=i: make_env(env_name, seed + config_env.n_env + 42 + i, mode = "test", 
                                                    visualize = config_env.visualize_env and i == 0) for i in range(config_env.n_env)], 
                             autoreset_mode= gym.vector.AutoresetMode.DISABLED)


    system.reproduce(seed)
    set_gpu_mode(torch.cuda.is_available(), FLAGS.device)

    ## now only use env and time as directory name
    run_name = f"{config_env.env_type}/{config_env.env_name}/"
    config_seq, _ = config_seq.name_fn(config_seq, env.get_attr("max_episode_steps")[0])
    max_training_steps = int(FLAGS.train_episodes * env.get_attr("max_episode_steps")[0])
    config_rl, _ = config_rl.name_fn(
        config_rl, env.get_attr("max_episode_steps")[0], max_training_steps
    )
    run_name = run_name + FLAGS.run_name 
    uid = f"_{system.now_str()}"
    run_name += uid


    log_dir = os.path.join(FLAGS.save_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    FLAGS.log_dir = log_dir

    # write flags to a txt
    key_flags = FLAGS.get_key_flags_for_module(argv[0])
    with open(os.path.join(log_dir, "flags.txt"), "w") as text_file:
        text_file.write("\n".join(f.serialize() for f in key_flags) + "\n")
    # write flags to pkl
    with open(os.path.join(log_dir, "flags.pkl"), "wb") as f:
        pickle.dump(FLAGS.flag_values_dict(), f)

    validate_flags(FLAGS)
    # start logger
    wandb.init(project = f"{env_name}", name = run_name, dir=log_dir, config = FLAGS.flag_values_dict())
    
    # start training
    learner = Learner(env, eval_env, FLAGS, config_rl, config_seq, config_env)
    learner.train()

def validate_flags(FLAGS):
    assert FLAGS.config_env.log_interval % FLAGS.config_env.n_env == 0 and FLAGS.config_env.eval_interval % FLAGS.config_env.n_env == 0, \
        "log_interval and eval_interval should be divisible by n_env."
    assert FLAGS.config_env.eval_episodes % FLAGS.config_env.n_env == 0, \
        "eval_episodes should be divisible by n_env."
    if FLAGS.config_rl.algo == "ppo":
        assert FLAGS.config_env.log_interval % FLAGS.batch_size == 0 and FLAGS.config_env.eval_interval % FLAGS.batch_size == 0, \
            "log_interval and eval_interval should be divisible by batch_size for PPO."
        assert FLAGS.batch_size % FLAGS.config_env.n_env == 0, \
            "batch_size for PPO should be divisible by n_env"
        if FLAGS.start_training > 0:
            FLAGS.start_training = 0
            print("start_training is set to 0, since PPO does not need it.")

if __name__ == "__main__":
    app.run(main)
