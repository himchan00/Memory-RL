from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def mate_rff_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)
    config.seq_model.max_seq_length = max_episode_steps + 1
    return config


def get_config():
    config = base_config()
    config.update_fn = mate_rff_update_fn

    # seq_model specific
    config.seq_model.name = "mate_rff"
    config.seq_model.hidden_size = 256          # must be even (RFF cos&sin estimator)
    config.seq_model.init_emb_zero = False      # if True, init_emb fixed to zeros (non-trainable buffer)

    # RFF knobs (consumed by RFFEmbedding in recurrent_head)
    config.seq_model.kernel = "gaussian"        # gaussian | laplace | matern | train

    return config
