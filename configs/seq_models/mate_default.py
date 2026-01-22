from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn

def mate_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)

    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: transition data starts from t=1

    return config


def get_config():
    config = ConfigDict()
    config.update_fn = mate_update_fn

    config.clip = True
    config.max_norm = 5.0

    # fed into Module
    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = True
    config.permutation_training = False  # whether to use permutation training (only for Mate)
    config.transition_dropout = 0.0  # dropout probability for transition embedding during training

    # seq_model specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "mate"
    config.seq_model.n_layer = 1 # 2 for metaworld, 1 for others
    config.seq_model.pdrop = 0.1
    config.seq_model.hidden_size = 128 # 256 for metaworld, 128 for mujoco & tmaze envs
    
    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"
    config.embedder.project_output = False

    return config
