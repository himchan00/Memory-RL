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
    config.seq_model.truncated_sampling = "window"  # subset | window

    config.seq_model.use_rff = False            # if True, last embedding layer is RFFEmbedding (kernel-mean MATE)
    config.seq_model.kernel = "gaussian"        # gaussian | laplace | matern (base measure; only when use_rff=True)
    config.seq_model.learn_kernel = "off"     # off | scale | linear | freq (kernel learning; only when use_rff=True)

    config.seq_model.learn_init_emb = True            # initial-memory prior: m_t=(w * init_emb + sum E)/(w + t)
    config.seq_model.use_ema_init_emb = False         # track init_emb as an EMA of valid training transition embeddings
    config.seq_model.ema_init_emb_beta = 5e-4
    config.seq_model.use_rollout_z_cache = False       # reconstruct omitted prefixes from cached rollout embeddings

    return config
