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

    # Entropy target as a fraction of the MAXIMUM achievable entropy
    # (Christodoulou 2019 uses 0.98). Discrete only. Applied to log(action_dim),
    # or to log(number of LEGAL actions) when invalid-action masking is on --
    # with masking, log(action_dim) is not reachable and the dual would drive
    # alpha up without bound chasing it.
    config.discrete_target_entropy_ratio = 0.98

    # Symbolic Alchemy only: mask actions for absent stones or used potions.
    # Must exist here to be overridable from the CLI; the mask is applied to
    # the policy logits BEFORE the softmax (see actor.CategoricalPolicy).
    config.mask_alchemy_invalid_actions = True
    config.mask_alchemy_no_op = True

    # Canonical-frame auxiliary supervision (see policies/models/aux_canon.py
    # and configs/rl/dqn_default.py for what each knob means). Shared with the
    # DQN agent through AuxCanonMixin, so the leak guard has one implementation.
    config.aux_canon_weight = 0.0

    # Contrastive version of the aux_canon label (policies/models/aux_cpc.py):
    # InfoNCE between the memory read-out and the label instead of regression.
    # Shares aux_canon_site / aux_canon_parts, so only the objective differs.
    config.aux_cpc_weight = 0.0        # 0 = off
    config.aux_cpc_tau = 0.1           # initial temperature; learned
    config.aux_cpc_proj_dim = 128
    config.aux_canon_parts = "potion"
    config.aux_canon_site = "memory_obs"

    # Use PopArt value normalization (https://arxiv.org/abs/1809.04474), following AMAGO (https://arxiv.org/abs/2411.11188).
    config.use_popart = True
    config.popart_beta = 5e-4
    config.popart_init_nu = 100.0

    return config
