from ml_collections import ConfigDict
from configs.seq_models.name_fns import name_fn


def get_config():
    config = ConfigDict()
    config.name_fn = name_fn

    config.is_markov = False

    config.sampled_seq_len = -1

    config.clip = False
    config.max_norm = 1.0
    config.use_l2_norm = False

    # fed into Module
    config.model = ConfigDict()
    config.model.obs_shortcut = True
    config.model.full_transition = True

    # seq_model specific
    config.model.seq_model_config = ConfigDict()
    config.model.seq_model_config.name = "hist"

    config.model.seq_model_config.hidden_size = 128
    config.model.seq_model_config.n_layer = 2

    # embedders
    config.model.observ_embedder = ConfigDict()
    config.model.observ_embedder.name = "mlp"
    config.model.observ_embedder.hidden_size = 32

    config.model.action_embedder = ConfigDict()
    config.model.action_embedder.name = "mlp"
    config.model.action_embedder.hidden_size = 16

    config.model.reward_embedder = ConfigDict()
    config.model.reward_embedder.name = "mlp"
    config.model.reward_embedder.hidden_size = 0

    return config
