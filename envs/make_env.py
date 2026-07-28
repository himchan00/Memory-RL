import gymnasium as gym
from .wrapper import oracleWrapper
from .metaworld import MLWrapper
from .k_episode_wrapper import KEpisodeWrapper

def make_env(
    env_name: str,
    seed: int,
    visualize: bool=False,
    **kwargs: dict,
) -> gym.Env:
    render_mode = "rgb_array" if visualize else None
    if env_name.startswith("ML"):
        # If the environment is from metaworld, use the MLWrapper class.
        env = MLWrapper(env_name, mode=kwargs["mode"], render_mode=render_mode,
                        max_episode_steps=kwargs.get("max_episode_steps"))
    else:
        # Check if the env is in gym.
        env = gym.make(env_name, render_mode=render_mode)
        env.max_episode_steps = getattr(
            env, "max_episode_steps", env.spec.max_episode_steps
        )
    # k-shot: concatenate k same-task attempts into one meta-episode. Applied
    # before the oracle wrapper so info["context"] / soft-reset flag flow through.
    k = int(kwargs.get("k", 1))
    if k > 1:
        env = KEpisodeWrapper(env, k)
    if kwargs.get("is_oracle", False):
        env = oracleWrapper(env)
    env.reset(seed=seed) # Set random seed
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


    return env
