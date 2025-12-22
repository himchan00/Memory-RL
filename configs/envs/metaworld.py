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
    config.terminate_after_success = True
    config.normalize_transitions = True # Whether to normalize observations, rewards, (NOT actions) for network input

    config.n_env = 64
    # eval_interval and log_interval and eval_episodes must be divisable by n_env. When training with PPO, we scale them by 10, as PPO uses more episodes.
    config.eval_interval = 256
    config.log_interval = 128
    config.eval_episodes = 64

    config.visualize_env = True
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval

    config.env_name = "ML10" # Possible choices: ["ML10", "ML45"]

    return config