from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

VEHICLE_SUBSETS = {
    "all":     list(range(29)),
    "basic":   [0, 9, 18, 27],          # RaceCar, StreetCar, Bus, TukTuk
    "race":    [0, 1, 2],               # RaceCar, FWDRaceCar, AWDRaceCar
    "diverse5": [0, 9, 18, 27, 6],      # + RaceCarLargeTrailer
}

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in VEHICLE_SUBSETS, (
        f"Invalid env_name: {env_name}. Choose from {list(VEHICLE_SUBSETS.keys())}."
    )
    vehicle_ids = VEHICLE_SUBSETS[env_name]
    registered_name = f"carl_vehicle_racing_{env_name}"

    register(
        registered_name,
        entry_point="envs.carl_vehicle_racing:CARLVehicleRacingWrapper",
        max_episode_steps=200,
        kwargs=dict(vehicle_ids=vehicle_ids, frame_skip=config.frame_skip),
    )

    del config.create_fn
    return config, registered_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "carl_vehicle_racing"
    config.horizon = "finite"
    config.terminate_after_success = False
    config.normalize_transitions = False  # CNN handles pixel normalization
    config.obs_backend = "memmap"
    config.obs_dtype = "uint8"
    config.frame_skip = 8
    config.n_env = 8
    config.eval_interval = 16
    config.log_interval = 16
    config.eval_episodes = 8

    config.visualize_env = True
    config.visualize_every = 2

    config.env_name = "all"

    return config