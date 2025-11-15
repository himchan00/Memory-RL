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
    config.transition_dropout_eval = True # whether to use transition dropout during sample/evaluation steps. 

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "hist"

    config.seq_model.out_act = "swish" # ex) "linear", "tanh"
    config.seq_model.temb_mode = "concat" # Only required when agg = "mean". One of ["none", "input", "output", "concat"]
    config.seq_model.hidden_size = 128

    config.seq_model.n_layer = 2
    # embedders (output_size is set to hidden_size of seq_model)
    config.transition_embedder = ConfigDict()
    config.transition_embedder.hidden_sizes = ()
    config.transition_embedder.norm = "none"
    config.transition_embedder.output_activation = "leakyrelu"

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.norm = "none"
    config.observ_embedder.output_activation = "leakyrelu"

    config.action_embedder = ConfigDict()
    config.action_embedder.hidden_sizes = ()
    config.action_embedder.norm = "none"
    config.action_embedder.output_activation = "leakyrelu"

    return config
