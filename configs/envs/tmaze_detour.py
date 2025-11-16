from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

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
    config.horizon = "finite" # finite or infinite
    config.terminate_after_success = True
    config.normalize_transitions = False # Whether to normalize observations, rewards, (NOT actions) for network input

    config.n_env = 32
    # eval_interval and log_interval and eval_episodes must be divisable by n_env
    config.eval_interval = 128
    config.log_interval = 64
    config.eval_episodes = 32

    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval

    config.env_name = 10 # Corridor length

    return config
