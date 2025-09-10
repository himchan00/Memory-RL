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
    config.max_norm = 1.0

    config.obs_shortcut = False
    config.full_transition = False
    
    # seq_model_config specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "gpt"

    config.seq_model.hidden_size = (
        128 
    )
    config.seq_model.n_layer = 1
    config.seq_model.n_head = 1
    config.seq_model.pdrop = 0.1 # Note: 0.1 is default
    config.seq_model.position_encoding = "sine"

    # embedders (output_size is set to hidden_size of seq_model)
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = ()
    config.transition_embedder.norm = "none"
    config.transition_embedder.dropout = 0
    config.transition_embedder.output_activation = "leakyrelu"

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.norm = "none"
    config.observ_embedder.dropout = 0
    config.observ_embedder.output_activation = "leakyrelu"

    config.action_embedder = ConfigDict()
    config.action_embedder.hidden_sizes = ()
    config.action_embedder.norm = "none"
    config.action_embedder.dropout = 0
    config.action_embedder.output_activation = "leakyrelu"

    return config
