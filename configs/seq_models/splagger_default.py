from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def get_config():
    config = base_config()
    config.update_fn = update_fn

    # seq_model specific
    config.seq_model.name = "splagger"
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 256
    config.seq_model.agg_type = "max"           # "max" (default) or "mean"

    return config
