from ml_collections import ConfigDict
from typing import Tuple
from configs.seq_models.name_fns import name_fn


def gpt_name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config, max_episode_steps)


    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: zero-prepend

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = gpt_name_fn

    config.clip = True
    config.max_norm = 2.0
    config.l2_norm = 1e-4

    config.obs_shortcut = False
    config.full_transition = False
    config.normalize_transitions = True
    
    # seq_model_config specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "gpt"

    config.seq_model.hidden_size = (
        256 
    )
    config.seq_model.n_layer = 3
    config.seq_model.n_head = 8
    config.seq_model.pdrop = 0 # Note: 0.1 is default
    config.seq_model.position_encoding = "sine"

    # embedders
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = (512, 256)
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
