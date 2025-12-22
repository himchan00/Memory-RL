from ml_collections import ConfigDict
from configs.rl.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.algo = "ppo"


    config.lr = 1e-4
    config.discount = 0.99
    config.lam = 0.95
    config.ppo_epochs = 10
    config.eps_clip = 0.2
    config.ent_coef = 0.001
    config.vf_coef = 0.5
    config.normalize_advantage = True

    config.continuous_action = True

    config.config_actor = ConfigDict()
    config.config_actor.hidden_dims = (256, 256)

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)


    return config
