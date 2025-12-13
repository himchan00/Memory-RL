from ml_collections import ConfigDict
from configs.rl.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.algo = "sac"

    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    config.temp_lr = 1e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dim = 256


    config.config_critic = ConfigDict()
    config.config_critic.hidden_dim = 256

    config.discount = 0.99
    config.tau = 0.001

    config.replay_buffer_num_episodes = 1e3

    config.update_temperature = True
    config.target_entropy = None

    return config
