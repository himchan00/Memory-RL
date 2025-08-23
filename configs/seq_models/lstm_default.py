from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.sampled_seq_len = -1

    config.clip = True
    config.max_norm = 1.0

    # fed into Module
    config.obs_shortcut = False
    config.full_transition = False

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "lstm"
    config.seq_model.hidden_size = 128
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1 # Note: 0.1 is default

    # embedders
    config.transition_embedder = ConfigDict()
    config.transition_embedder.norm = "layer"
    config.transition_embedder.dropout = 0.1

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.output_size = 64
    config.observ_embedder.norm = "layer"
    config.observ_embedder.dropout = 0.1

    return config
