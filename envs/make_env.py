import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from .metaworld import ml_env

def make_env(
    env_name: str,
    seed: int,
    visualize: bool=False,
    **kwargs: dict,
) -> gym.Env:
    render_mode = "rgb_array" if visualize else None
    if env_name.startswith("ML"):
        # If the environment is from metaworld, use the ml_env class.
        env = ml_env(env_name, mode=kwargs["mode"], render_mode=render_mode)
        env.max_episode_steps = env.max_path_length # 500 for now
    else:
        # Check if the env is in gym.
        env = gym.make(env_name, render_mode=render_mode)
        env.max_episode_steps = getattr(
            env, "max_episode_steps", env.spec.max_episode_steps
        )

    # if isinstance(env.action_space, gym.spaces.Box): # Commented temporarily because it causes issues with some environments.
    #     print(env.max_episode_steps)
    #     env = RescaleAction(env, -1.0, 1.0)
    #     print(env.max_episode_steps)

    env.reset(seed=seed) # Set random seed
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


    return env
