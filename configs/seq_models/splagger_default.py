from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.clip = True
    config.max_norm = 5.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = False

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "splagger"
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 256
    config.seq_model.agg_type = "max"  # "max" (default) or "mean"

    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"


    return config
