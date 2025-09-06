from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

env_name_fn = lambda l: f"tmaze_active_T-{l}"


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    length = config.env_name
    env_name = env_name_fn(length)
    register(
        env_name,
        entry_point="envs.tmaze:TMazeClassicActive",
        kwargs=dict(
            corridor_length=length,
            penalty=-1.0 / length,  # NOTE: \sum_{t=1}^T -1/T = -1
        ),
        max_episode_steps=length + 2 * 1 + 1,  # NOTE: has to define it here
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "tmaze_active"
    config.horizon = "finite" # finite or infinite
    config.terminate_after_success = True
    config.add_time = True # Whether to add time step to observation

    config.n_env = 32
    # eval_interval and log_interval and eval_episodes must be divisable by n_env
    config.eval_interval = 128
    config.log_interval = 64
    config.eval_episodes = 32

    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval

    config.env_name = 10 # Corridor length

    return config
