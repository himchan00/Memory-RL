from ml_collections import ConfigDict
from typing import Tuple

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in ["ML10", "ML45"], f"Invalid environment name: {env_name}. Choose from ['ML10', 'ML45']."

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "Metaworld"
    config.horizon = "finite" # finite or infinite

    config.n_env = 32
    # eval_interval and log_interval must be divisable by n_env
    config.eval_interval = 128
    config.log_interval = 64
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10
    config.visualize_env = True

    config.env_name = "ML10" # Possible choices: ["ML10", "ML45"]

    return config