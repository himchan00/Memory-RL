from ml_collections import ConfigDict
from configs.seq_models.update_fns import update_fn

def gpt_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)
    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: zero-prepend

    return config


def get_config():
    config = ConfigDict()
    config.update_fn = gpt_update_fn

    config.clip = True
    config.max_norm = 5.0

    config.obs_shortcut = True
    config.full_transition = True
    config.project_output = False
    
    # seq_model_config specific
    config.seq_model = ConfigDict()
    config.seq_model.name = "gpt"

    config.seq_model.n_layer = 1
    config.seq_model.n_head = 1
    config.seq_model.pdrop = 0.1
    config.seq_model.position_encoding = "sine"  # one of ["sine", "learned", "none"]
    config.seq_model.hidden_size = 128 # 128 for mujoco envs, 32 for tmaze envs

    #(transition, observation, action, context) embedder configs
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"
    config.embedder.project_output = False

    return config
