from ml_collections import ConfigDict
from configs.rl.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.algo = "sac"

    config.actor_lr = 1e-4
    config.critic_lr = 1e-4
    config.temp_lr = 1e-4

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (512, 512)


    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (512, 512)

    config.discount = 0.99
    config.tau = 0.001

    config.replay_buffer_num_episodes = 1e4

    config.update_temperature = True
    config.target_entropy = None

    # Use PopArt value normalization (https://arxiv.org/abs/1809.04474), following AMAGO (https://arxiv.org/abs/2411.11188).
    config.use_popart = True
    config.popart_beta = 5e-4
    config.popart_init_nu = 100.0

    return config
