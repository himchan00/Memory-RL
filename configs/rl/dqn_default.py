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
    config.tau = 0.003
    config.schedule_end = 0.1  # at least good for TMaze-like envs

    config.replay_buffer_num_episodes = 1e4

    # Use PopArt value normalization (https://arxiv.org/abs/1809.04474), following AMAGO (https://arxiv.org/abs/2411.11188).
    config.use_popart = False
    config.popart_beta = 5e-4
    config.popart_init_nu = 100.0

    # Symbolic Alchemy only: mask actions for absent stones or potions.
    config.mask_alchemy_invalid_actions = False

    # Symbolic Alchemy only: replace the flat 40-way critic with a dueling +
    # factored head over NO_OP + stone(3) x target(13). Strictly no less
    # expressive than the flat head; adds per-stone and per-target advantage
    # terms so a fact about one stone is shared across its 13 targets.
    # See policies/models/action_heads.py.
    config.factored_action_head = False

    return config
