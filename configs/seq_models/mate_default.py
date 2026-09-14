from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def mate_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)

    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: transition data starts from t=1

    return config


def get_config():
    config = base_config()
    config.update_fn = mate_update_fn
    
    # MATE-specific defaults
    config.obs_shortcut = True

    # seq_model specific
    config.seq_model.name = "mate"
    config.seq_model.n_layer = 1                # 2 for metaworld, 1 for others
    config.seq_model.hidden_size = 256

    config.seq_model.use_rff = False            # if True, last embedding layer is RFFEmbedding (kernel-mean MATE)
    config.seq_model.kernel = "gaussian"        # gaussian | laplace | matern (base measure; only when use_rff=True)
    config.seq_model.learn_kernel = "off"     # off | scale | linear | freq (kernel learning; only when use_rff=True)

    config.seq_model.learn_init_emb = True            # learnable init prior: m_t=(init_emb + sum E)/(w + t), w=exp(log_init_weight); else (sum E)/t
    # Initial w in m_t = (init_emb + sum z_i) / (w + t). It sets how far the
    # memory's magnitude swings across an episode: (w+T)/(w+1). At 1.0 that is
    # 100x over 200 steps; trained runs climb to 4-38 on their own, and climb
    # LESS when the loss is reweighted toward the part memory is for.
    config.seq_model.init_weight = 1.0

    # Per-transition gate w_i in (0,1): m_t = (init_emb + sum w_i z_i) / (w + sum w_i).
    # Permutation-invariant (the gate reads only its own transition), so it can
    # down-weight the ~85% of Alchemy steps that carry no chemistry evidence.
    config.seq_model.use_gate = False

    return config
