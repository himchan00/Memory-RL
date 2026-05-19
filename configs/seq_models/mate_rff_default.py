from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn


def mate_rff_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)
    config.seq_model.max_seq_length = max_episode_steps + 1
    return config


def get_config():
    config = ConfigDict()
    config.update_fn = mate_rff_update_fn

    config.clip = True
    config.max_norm = 1.0
    config.compile = False
    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = False # Use mean aggregation for mate_rff

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "mate_rff"
    config.seq_model.hidden_size = 256  # must be even (RFF cos&sin estimator)
    config.seq_model.init_emb_zero = False  # if True, init_emb fixed to zeros (non-trainable buffer)

    # RFF knobs (consumed by RFFEmbedding in recurrent_head)
    config.seq_model.kernel = "gaussian"  # gaussian | laplace | matern | riesz

    # (transition, observation, action, context) embedder configs.
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.normalize_inputs = True
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"

    return config
