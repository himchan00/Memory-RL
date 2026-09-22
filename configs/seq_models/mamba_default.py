from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def get_config():
    config = base_config()
    config.update_fn = update_fn

    # seq_model specific (AMAGO-style Mamba blocks)
    config.seq_model.name = "mamba"
    config.seq_model.n_layer = 1
    config.seq_model.hidden_size = 256
    config.seq_model.d_state = 16
    config.seq_model.d_conv = 4
    config.seq_model.expand = 2

    return config
