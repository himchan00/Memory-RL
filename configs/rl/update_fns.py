from ml_collections import ConfigDict

def update_fn(config: ConfigDict, *args) -> ConfigDict:
    """
    A function to update the configuration dictionary based on input parameters
    """
    del config.update_fn
    return config
