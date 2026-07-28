import gymnasium as gym
import numpy as np


class KEpisodeWrapper(gym.Wrapper):
    """Concatenate ``k`` same-task attempts into one meta-episode (k-shot / RL^2).

    Mirrors amago's ``KEpisode*`` wrappers. The underlying task/context is held
    fixed across the ``k`` attempts; the policy has ``k`` tries to adapt. Each
    attempt runs for the inner env's full horizon (an attempt boundary is the
    inner env's ``truncated`` flag), so the meta-episode always has a fixed length
    ``k * inner_max_episode_steps`` — required by the fixed-size RolloutBuffer.

    - A soft-reset signal is appended to the observation as one extra channel:
      ``1.0`` on the first step of each attempt (including the very first), else
      ``0.0``. This tells the policy a new attempt just began.
    - On an attempt boundary the inner env is reset via
      ``reset(options={"keep_context": True})`` so the task/context is preserved.
      Envs opt in to this protocol (mujoco natively; tmaze/metaworld extended).
    - The meta-episode is truncation-based (RL^2): per-attempt termination is not
      propagated as a done, so the value function bootstraps across attempt
      boundaries. ``truncated`` is emitted only after the k-th attempt.

    Note: this relies on the same no-early-termination assumption the rest of the
    codebase makes (rollouts run until ``truncated``); attempts are expected to
    reach their full horizon (e.g. mujoco ``terminate_when_unhealthy=False``).
    """

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        assert k >= 1, "k must be >= 1"
        self.k = int(k)
        inner_max = getattr(env, "max_episode_steps", None)
        assert inner_max is not None, "inner env must expose max_episode_steps for k-shot"
        self.inner_max_episode_steps = int(inner_max)
        self.max_episode_steps = self.inner_max_episode_steps * self.k

        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box), "KEpisodeWrapper supports Box obs only"
        low = np.concatenate([np.asarray(obs_space.low, dtype=np.float64), [0.0]])
        high = np.concatenate([np.asarray(obs_space.high, dtype=np.float64), [1.0]])
        self.observation_space = gym.spaces.Box(
            low=low.astype(obs_space.dtype), high=high.astype(obs_space.dtype),
            dtype=obs_space.dtype,
        )

        self._current_k = 0

    def _augment(self, obs, soft_reset: bool):
        obs = np.asarray(obs)
        flag = np.array([1.0 if soft_reset else 0.0], dtype=obs.dtype)
        return np.concatenate([obs, flag])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_k = 0
        info = {**info, "soft_reset": True}
        return self._augment(obs, soft_reset=True), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        soft_reset = False
        meta_truncated = False
        # Attempt boundary = inner truncation (fixed per-attempt horizon).
        if truncated:
            self._current_k += 1
            if self._current_k >= self.k:
                meta_truncated = True
            else:
                # Same task/context, fresh attempt.
                obs, reset_info = self.env.reset(options={"keep_context": True})
                info = {**info, **reset_info}
                soft_reset = True
        info["soft_reset"] = soft_reset
        # RL^2: never propagate per-attempt termination as a meta-done; the
        # meta-episode ends only via truncation after the k-th attempt.
        return self._augment(obs, soft_reset=soft_reset), reward, False, meta_truncated, info
