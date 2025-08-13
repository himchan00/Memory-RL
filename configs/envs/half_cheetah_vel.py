from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = "C-vel-v0"
    register(
        env_name,
        entry_point="envs.mujoco:HalfCheetahVelEnv",
        max_episode_steps=200
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "half_cheetah_vel"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 50
    config.log_interval = 10
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10

    config.env_name = "C-vel-v0"

    return config