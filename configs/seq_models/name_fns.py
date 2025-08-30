from ml_collections import ConfigDict
from typing import Tuple


def name_fn(config: ConfigDict, max_episode_steps: int) -> Tuple[ConfigDict, str]:
    name = ""

    name += f"{config.seq_model.name}-len-{max_episode_steps}/"

    del config.name_fn
    return config, name
