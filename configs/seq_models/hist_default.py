from ml_collections import ConfigDict
from typing import Tuple
from configs.seq_models.name_fns import name_fn

def hist_name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config, max_episode_steps)

    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: transition data starts from t=1

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = hist_name_fn

    config.clip = True
    config.max_norm = 5.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.transition_dropout_range = (0.0, 0.0)

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "hist"

    config.seq_model.out_act = "swish" # ex) "linear", "tanh"
    config.seq_model.temb_mode = "none" # One of ["none", "add", "concat"]
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.norm = "none" # One of ["none", "layer", "batch"]
    config.seq_model.hidden_size = 128 # 128 for mujoco envs, 32 for tmaze envs

    config.seq_model.init_emb_mode = "obs" # One of ["obs", "parameter", "zero"]
    
    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"

    return config
