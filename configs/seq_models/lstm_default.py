from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.clip = True
    config.max_norm = 5.0
    config.l2_norm = 1e-4

    # fed into Module
    config.obs_shortcut = False
    config.full_transition = False
    config.normalize_transitions = True

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "lstm"
    config.seq_model.hidden_size = 128
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1 # Note: 0.1 is default

    # embedders (output_size is set to hidden_size of seq_model)
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = ()
    config.transition_embedder.norm = "layer"
    config.transition_embedder.norm_mode = "final"
    config.transition_embedder.dropout = 0

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.output_size = 64
    config.observ_embedder.norm = "layer"
    config.observ_embedder.norm_mode = "final"
    config.observ_embedder.dropout = 0

    return config
