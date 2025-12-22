from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn


def get_config():
    config = ConfigDict()
    config.update_fn = update_fn

    config.clip = True
    config.max_norm = 3.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = True

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "lstm"
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 128 # 128 for mujoco envs, 32 for tmaze envs

    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"
    config.embedder.project_output = True


    return config
