import gymnasium as gym
from .metaworld import MLWrapper

def make_env(
    env_name: str,
    seed: int,
    visualize: bool=False,
    **kwargs: dict,
) -> gym.Env:
    render_mode = "rgb_array" if visualize else None
    if env_name.startswith("ML"):
        # If the environment is from metaworld, use the MLWrapper class.
        env = MLWrapper(env_name, mode=kwargs["mode"], render_mode=render_mode)
    else:
        # Check if the env is in gym.
        env = gym.make(env_name, render_mode=render_mode)
        env.max_episode_steps = getattr(
            env, "max_episode_steps", env.spec.max_episode_steps
        )


    env.reset(seed=seed) # Set random seed
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


    return env
