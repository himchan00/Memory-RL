from ml_collections import ConfigDict
from typing import Tuple
from configs.seq_models.name_fns import name_fn

def hist_name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    config, name = name_fn(config, max_episode_steps)

    config.model.seq_model_config.hidden_size = 0
    if config.model.observ_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.observ_embedder.hidden_size
        )
        if config.model.full_transition:
            config.model.seq_model_config.hidden_size += (
                config.model.observ_embedder.hidden_size
            )
    if config.model.action_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.action_embedder.hidden_size
        )
    if config.model.reward_embedder is not None:
        config.model.seq_model_config.hidden_size += (
            config.model.reward_embedder.hidden_size
        )

    config.model.seq_model_config.max_seq_length = (
        config.sampled_seq_len + 1
    )  # NOTE: transition data starts from t=1

    return config, name


def get_config():
    config = ConfigDict()
    config.name_fn = hist_name_fn

    config.is_markov = False

    config.sampled_seq_len = -1

    config.clip = True
    config.max_norm = 1.0

    # fed into Module
    config.model = ConfigDict()
    config.model.obs_shortcut = True
    config.model.full_transition = True

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "hist"

    # This is current default config for mean agg
    config.model.seq_model_config.agg = "mean" # assert agg in ["sum", "logsumexp", "mean"]
    config.model.seq_model_config.out_act = "linear" # ex) "linear", "tanh"
    config.model.seq_model_config.temb_mode = "output" # Only required when agg = "mean". One of ["none", "input", "output"]

    # This is current default config for sum agg
    # config.model.seq_model_config.agg = "sum" # assert agg in ["sum", "logsumexp", "mean"]
    # config.model.seq_model_config.out_act = "tanh" # ex) "linear", "tanh"
    # config.model.hyp_emb = True # Only required when agg = "sum"


    config.model.seq_model_config.hidden_size = (
        128  # NOTE: will be overwritten by name_fn
    )
    config.model.seq_model_config.n_layer = 2
    config.model.seq_model_config.pdrop = 0.1 # 0.1 is default

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_size = 64

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_size = 48

    config.model.reward_embedder = ConfigDict()
    config.model.reward_embedder.name = "mlp"
    config.model.reward_embedder.hidden_size = 16

    return config
