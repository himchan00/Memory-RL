import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, VectorEnv  # VectorEnv used for isinstance checks
from .metaworld import ml_env

def make_env(
    env_name: str,
    seed: int,
    mode: str = "train",
    multi_env: int | None = None,
    render_mode: str | None = None,
):
    """
    - Multi-env: SyncVectorEnv of headless single envs (render_mode ignored).
    - Single env: forwards render_mode (use 'rgb_array' for GIF eval).
    """
    # ----- Metaworld path (kept single env) -----
    if env_name.startswith("ML"):
        env = ml_env(env_name, mode=mode)
        env.max_episode_steps = getattr(env, "max_path_length", 500)
        env.reset(seed=seed)
        if hasattr(env.action_space, "seed"): env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"): env.observation_space.seed(seed)
        return env

    # ----- Vectorized training -----
    if isinstance(multi_env, int) and multi_env > 1:
        def _thunk():
            # headless for speed
            return gym.make(env_name)

        env = SyncVectorEnv([_thunk for _ in range(multi_env)])

        # best-effort episode length on wrapper
        try:
            spec = gym.spec(env_name)
            max_steps = getattr(spec, "max_episode_steps", None)
        except Exception:
            max_steps = None
        if max_steps is not None:
            setattr(env, "max_episode_steps", max_steps)

        # proper vector seeding: list of seeds
        env.reset(seed=[seed + i for i in range(multi_env)])
        return env

    # ----- Single env (default & for eval/GIFs) -----
    env = gym.make(env_name, render_mode=render_mode)
    env.max_episode_steps = getattr(
        env, "max_episode_steps",
        getattr(getattr(env, "spec", None), "max_episode_steps", None)
    )
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"): env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"): env.observation_space.seed(seed)
    return env

