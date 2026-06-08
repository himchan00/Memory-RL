from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

from configs.envs.common import base_config

env_name_fn = lambda l: f"tmaze_passive_T-{l}"


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.tmaze:TMazeClassicPassive",
        kwargs=dict(
            corridor_length=length,
            penalty=-1.0 / (length+1),  # NOTE: \sum_{t=1}^T -1/T = -1
        ),
        max_episode_steps=length + 1,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = base_config()
    config.create_fn = create_fn

    config.env_type = "tmaze_passive"
    config.horizon = "finite"  # finite or infinite

    config.env_name = 10  # Corridor length

    return config
