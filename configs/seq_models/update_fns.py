from ml_collections import ConfigDict

def update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    """
    A function to update the configuration dictionary based on input parameters (ex. max_episode_steps).
    """
    # Full meta-episode length (+1 for the t=-1 dummy). Available to every seq model
    # (e.g. sizes the use_pe positional-encoding table). mate/gpt re-set the same value.
    config.seq_model.max_seq_length = max_episode_steps + 1
    del config.update_fn
    return config
