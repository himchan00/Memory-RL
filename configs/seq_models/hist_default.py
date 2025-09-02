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

    config.clip = False
    config.max_norm = 1.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.normalize_transitions = True

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "hist"

    # This is current default config for mean agg
    # config.seq_model.agg = "mean" # assert agg in ["sum", "logsumexp", "mean"]
    # config.seq_model.out_act = "linear" # ex) "linear", "tanh"
    # config.seq_model.temb_mode = "concat" # Only required when agg = "mean". One of ["none", "input", "output", "concat"]
    # config.seq_model.temb_size = 128 # Only used when temb_mode = "concat"

    # This is current default config for sum agg
    # config.seq_model.agg = "sum" # assert agg in ["sum", "logsumexp", "mean"]
    # config.seq_model.out_act = "linear" # ex) "linear", "tanh"
    # config.hyp_emb = True # Only required when agg = "sum"

    # This is current default config for gaussian agg
    config.seq_model.agg = "gaussian" # assert agg in ["sum", "logsumexp", "mean"]
    config.seq_model.out_act = "linear" # ex) "linear", "tanh"

    config.seq_model.hidden_size = (
        128 
    )
    config.seq_model.n_layer = 1
    config.seq_model.pdrop = 0.1 # 0.1 is default
    config.seq_model.norm = "none" # one of "none", "layer", "spectral"

    # embedders
    config.transition_embedder = ConfigDict()
    config.transition_embedder.norm = "none"
    config.transition_embedder.dropout = 0.1

    config.observ_embedder = ConfigDict()
    config.observ_embedder.hidden_sizes = ()
    config.observ_embedder.output_size = 64
    config.observ_embedder.norm = "none"
    config.observ_embedder.dropout = 0.1

    return config
