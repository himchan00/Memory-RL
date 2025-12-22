from configs.rl.update_fns import update_fn
from ml_collections import ConfigDict

    
def dqn_update_fn(
    config: ConfigDict, max_episode_steps: int, max_training_steps: int
) -> ConfigDict:
    config = update_fn(config)
    # set eps = 1/T, so that the asymptotic prob to
    # sample fully exploited trajectory during exploration is
    # (1-1/T)^T = 1/e
    config.init_eps = 1.0
    config.end_eps = 1.0 / max_episode_steps
    config.schedule_steps = config.schedule_end * max_training_steps

    return config


def get_config():
    config = ConfigDict()
    config.update_fn = dqn_update_fn

    config.algo = "dqn"

    config.critic_lr = 1e-4

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.005
    config.schedule_end = 0.1  # at least good for TMaze-like envs

    config.replay_buffer_num_episodes = 1e3

    return config
