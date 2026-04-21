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
    config.config_actor.hidden_dims = (256, 256)  # smaller than default (512,512)
    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)
    config.discount = 0.99
    config.tau = 0.001
    config.replay_buffer_num_episodes = 10000  # 1001 steps * 27648 obs_dim * float32 * 2 tensors ≈ 10.3 GiB for 50 eps
    config.update_temperature = True
    config.target_entropy = None
    return config