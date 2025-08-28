from ml_collections import ConfigDict
from typing import Tuple
from configs.envs.terminal_fns import finite_horizon_terminal

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in ["ML10", "ML45"], f"Invalid environment name: {env_name}. Choose from ['ML10', 'ML45']."

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "Metaworld"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.log_interval = 50
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10
    config.visualize_env = True

    config.env_name = "ML10" # Possible choices: ["ML10", "ML45"]

    return config