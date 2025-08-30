from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

env_name_fn = lambda l: f"T-{l}-v0"


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.tmaze:TMazeDetour",
        kwargs=dict(
            corridor_length=length,
        ),
        max_episode_steps=length + 3 + 1,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "tmaze_detour"
    config.terminal_fn = finite_horizon_terminal

    config.n_env = 8
    config.eval_interval = 50
    config.log_interval = 10
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10

    config.env_name = 10 # Corridor length

    return config
