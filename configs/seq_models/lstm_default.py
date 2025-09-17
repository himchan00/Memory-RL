from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.clip = True
    config.max_norm = 0.1

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "lstm"
    config.seq_model.n_layer = 2
    config.seq_model.pdrop = 0

    # embedders (output_size is set to hidden_size of seq_model)
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = ()
    config.transition_embedder.norm = "none"
    config.transition_embedder.output_activation = "leakyrelu"

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.norm = "none"
    config.observ_embedder.output_activation = "leakyrelu"

    config.action_embedder = ConfigDict()
    config.action_embedder.hidden_sizes = ()
    config.action_embedder.norm = "none"
    config.action_embedder.output_activation = "leakyrelu"


    return config
