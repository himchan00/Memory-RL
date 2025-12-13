from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.clip = True
    config.max_norm = 5.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = False

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "markov"
    config.seq_model.hidden_size = 0
    config.seq_model.context_emb_dim = 128 # 128 for mujoco envs, 32 for tmaze envs
    config.seq_model.is_oracle = False # If True, use oracle Markov model that takes context embedding as input

    # embedders (output_size is set to hidden_size of seq_model)
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = ()
    config.transition_embedder.norm = "none"
    config.transition_embedder.output_activation = "leakyrelu"

    return config
