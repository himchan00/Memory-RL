from ml_collections import ConfigDict
from typing import Tuple

from configs.envs.common import base_config


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in ["ML10", "ML45"], f"Invalid environment name: {env_name}. Choose from ['ML10', 'ML45']."

    del config.create_fn
    return config, env_name


def get_config():
    config = base_config()
    config.create_fn = create_fn

    config.env_type = "Metaworld"
    config.horizon = "infinite"  # finite or infinite

    config.env_name = "ML10"  # Possible choices: ["ML10", "ML45"]
    config.max_episode_steps = 500  # metaworld default (500)

    return config
