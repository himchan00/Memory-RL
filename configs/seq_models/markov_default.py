from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.clip = True
    config.max_norm = 5.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = False
    config.project_output = False

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "markov"
    # Note: Markov model does not have a hidden state, but we set hidden_size to define the observation embedding size.
    config.seq_model.hidden_size = 256 # 256 for metaworld, 128 for mujoco & tmaze envs
    config.seq_model.is_oracle = False # If True, use oracle Markov model that takes context embedding as input

    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"
    config.embedder.project_output = False


    return config
