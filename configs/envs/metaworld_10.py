from ml_collections import ConfigDict
from typing import Tuple
from configs.envs.terminal_fns import finite_horizon_terminal

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = "ML10"

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "ML10"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.log_interval = 50
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10

    config.env_name = "ML10"

    return config