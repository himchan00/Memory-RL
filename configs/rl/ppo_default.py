from ml_collections import ConfigDict
from configs.rl.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.algo = "ppo"

    # default values from stable baselines 3, only adjust ent_coef
    config.lr = 3e-4
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
