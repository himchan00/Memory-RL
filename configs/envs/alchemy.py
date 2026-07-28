from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

from configs.envs.common import base_config

# Short alias -> dm_alchemy symbolic level_name. dm_alchemy parses the level_name
# by substring (perceptual_mapping_randomized / rotation / random_bottleneck |
# bottleneck1..3), so there is no fixed registry — these are the useful combos.
LEVELS = {
    "rotation_random_bottleneck":
        "perceptual_mapping_randomized_with_rotation_and_random_bottleneck",
    "random_bottleneck":
        "perceptual_mapping_randomized_with_random_bottleneck",
    "rotation_no_bottleneck":
        "perceptual_mapping_randomized_with_rotation",
    "no_bottleneck":
        "perceptual_mapping_randomized",
    "rotation_bottleneck1":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck1",
    "rotation_bottleneck2":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck2",
    "rotation_bottleneck3":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck3",
    "all_fixed":
        "all_fixed_with_no_bottleneck",
    "all_fixed_rotation":
        "all_fixed_with_rotation",
}


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in LEVELS, (
        f"Invalid alchemy env_name: {env_name}. Choose from {list(LEVELS.keys())}.")
    register(
        env_name,
        entry_point="envs.alchemy:SymbolicAlchemyEnv",
        max_episode_steps=config.num_trials * config.max_steps_per_trial,
        kwargs=dict(
            level_name=LEVELS[env_name],
            num_trials=config.num_trials,
            max_steps_per_trial=config.max_steps_per_trial,
            observe_used=config.observe_used,
            add_trial_flag=config.add_trial_flag,
        ),
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = base_config()
    config.create_fn = create_fn

    config.env_type = "alchemy"
    config.horizon = "finite"  # finite or infinite

    # Symbolic Alchemy meta-episode: `num_trials` trials share one hidden
    # chemistry. Run with --k 1 (the multi-trial structure is native; do NOT use
    # KEpisodeWrapper). The learner reads num_trials to log per-attempt
    # adaptation curves (return_attempt_0..N).
    config.env_name = "rotation_random_bottleneck"  # see LEVELS for choices
    config.num_trials = 10
    config.max_steps_per_trial = 20
    config.observe_used = True
    config.add_trial_flag = True

    return config
