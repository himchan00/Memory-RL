"""Shared defaults for all `configs/envs/*.py` configs.

Each specific config calls `base_config()`, then sets `create_fn`, `env_type`,
`env_name`, `horizon`, and overrides anything env-specific (e.g. carl flips
`terminate_after_success` to False and `n_env` to 32).

Keep this file in sync with how `main.py` and `policies/learner.py` consume
`config_env.*` keys (n_env, eval_interval, log_interval, eval_episodes,
visualize_env, visualize_every, horizon, terminate_after_success,
max_episode_steps).
"""
from ml_collections import ConfigDict


def base_config() -> ConfigDict:
    """Common `config_env.*` defaults shared by every environment."""
    config = ConfigDict()

    # rollout settings.
    # eval_interval, log_interval, and eval_episodes must all be divisible by n_env.
    config.n_env = 64
    config.eval_interval = 256
    config.log_interval = 128
    config.eval_episodes = 64

    # episode termination
    config.terminate_after_success = True

    # visualization
    config.visualize_env = False
    config.visualize_every = 5  # visualize_interval = visualize_every * log_interval

    return config
