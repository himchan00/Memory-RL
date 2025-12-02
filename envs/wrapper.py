import gymnasium as gym
import numpy as np

class oracleWrapper(gym.Wrapper):
    """Gym wrapper to provide oracle context information in the observation."""

    def __init__(self, env: gym.Env):
        _, info = env.reset()
        context = info.get("context", None)
        assert context is not None, "Environment does not provide context info in info dict"
        self.context_dim = len(context)

        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Box) and context is not None:
            orig_space = env.observation_space
            low_context = np.full((self.context_dim,), -np.inf, dtype=orig_space.dtype)
            high_context = np.full((self.context_dim,), np.inf, dtype=orig_space.dtype)

            low = np.concatenate([orig_space.low, low_context], axis=-1)
            high = np.concatenate([orig_space.high, high_context], axis=-1)

            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=orig_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.concatenate([obs, info["context"]], axis=-1)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate([obs, info["context"]], axis=-1)
        return obs, reward, terminated, truncated, info