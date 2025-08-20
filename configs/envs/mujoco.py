from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register
from configs.envs.terminal_fns import finite_horizon_terminal

ENTRY_POINTS = {"cheetah-vel": "envs.mujoco:HalfCheetahVelEnv", "ant-dir": "envs.mujoco:AntDirEnv", 
                "hopper-param": "envs.mujoco:HopperRandParamsEnv", "walker-param": "envs.mujoco:Walker2DRandParamsEnv"}

def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in ENTRY_POINTS, f"Invalid environment name: {env_name}. Choose from {list(ENTRY_POINTS.keys())}."
    entry_point = ENTRY_POINTS[env_name]
    register(
        env_name,
        entry_point=entry_point,
        max_episode_steps=200
    )

    del config.create_fn
    return config, env_name


def get_config():
    config = ConfigDict()
    config.create_fn = create_fn

    config.env_type = "mujoco"
    config.terminal_fn = finite_horizon_terminal

    config.eval_interval = 100
    config.log_interval = 50
    config.visualize_every = 5 # visualize_interval = visualize_every * log_interval
    config.eval_episodes = 10

    config.env_name = "cheetah-vel" # Possible choices: ["cheetah-vel", "cheetah-dir", "ant-dir", "hopper-param", "walker-param"]

    return config