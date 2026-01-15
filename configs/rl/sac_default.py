from ml_collections import ConfigDict
from configs.rl.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.algo = "sac"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (512, 512)


    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (512, 512)

    config.discount = 0.99
    config.tau = 0.001

    config.replay_buffer_num_episodes = 1e4

    config.update_temperature = True
    config.target_entropy = None

    return config
