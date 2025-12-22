from ml_collections import ConfigDict

def update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    """
    A function to update the configuration dictionary based on input parameters (ex. max_episode_steps).
    """
    del config.update_fn
    return config
