from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def get_config():
    config = base_config()
    config.update_fn = update_fn

    # markov-specific: obs shortcut is neccessary, as they do not use transition history.
    config.obs_shortcut = True

    # seq_model specific
    config.seq_model.name = "markov"
    # Note: Markov model does not have a hidden state, but we set hidden_size to define the observation embedding size.
    config.seq_model.n_layer = 1                # 2 for metaworld, 1 for others
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 256          # 256 for metaworld, 128 for mujoco & tmaze envs
    config.seq_model.is_oracle = False          # If True, use oracle Markov model that takes context embedding as input

    return config
