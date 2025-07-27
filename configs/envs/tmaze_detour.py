from ml_collections import ConfigDict
from typing import Tuple
from gym.envs.registration import register
from configs.envs.terminal_fns import infinite_horizon_terminal

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
    config.terminal_fn = infinite_horizon_terminal

    config.eval_interval = 10
    config.save_interval = 10
    config.eval_episodes = 10

    # [1, 2, 5, 10, 30, 50, 100, 300, 500, 1000]
    config.env_name = 10

    return config
