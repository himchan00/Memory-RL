import gymnasium as gym
import numpy as np
from envs.carl.carl_vehicle_racing import CustomCarRacing, PARKING_GARAGE

class CARLVehicleRacingWrapper(gym.Env):
    """
    Memory-RL compatible wrapper for CARL Vehicle Racing.

    - Stores raw 96x96x3 images as flattened uint8->float32 vectors
    - Randomly samples vehicle type each episode
    - Returns context (vehicle_id) in info dict
    - Observation space: Box(27648,) float32 [0, 1]
    - Action space: Box(3,) float32 [-1, 1]
    """

    IMAGE_SHAPE = (3, 96, 96)  # C, H, W (for CNN encoder)
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, vehicle_ids=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        if vehicle_ids is None:
            vehicle_ids = [0]  # default: RaceCar only
        self.vehicle_ids = vehicle_ids
        self.vehicle_classes = [PARKING_GARAGE[vid] for vid in vehicle_ids]

        self._env = CustomCarRacing(
            vehicle_class=self.vehicle_classes[0],
            verbose=False,
            render_mode=render_mode,
        )

        # Obs: flattened image (stored as float32 for buffer compatibility)
        self.obs_dim = 96 * 96 * 3  # 27648
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action: SAC outputs tanh actions in [-1,1]^d, but CarRacing expects
        # steering in [-1,1], gas in [0,1], brake in [0,1].
        # Expose symmetric [-1,1]^3 to the agent; rescale in step().
        self._real_action_space = self._env.action_space
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._action_low = self._real_action_space.low    # [-1, 0, 0]
        self._action_high = self._real_action_space.high  # [1, 1, 1]
        self.max_episode_steps = 1000
        self._current_vehicle_id = vehicle_ids[0]

    def reset(self, seed=None, options=None, **kwargs):
        # Sample random vehicle
        idx = np.random.randint(len(self.vehicle_ids))
        self._current_vehicle_id = self.vehicle_ids[idx]
        self._env.vehicle_class = self.vehicle_classes[idx]

        obs, info = self._env.reset(seed=seed, options=options)
        obs_flat = obs.astype(np.float32).flatten()  # (27648,)
        info["context"] = np.array([self._current_vehicle_id], dtype=np.float32)
        return obs_flat, info

    def step(self, action):
        # Rescale from [-1,1] to each dimension's actual bounds
        action = (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low
        obs, reward, terminated, truncated, info = self._env.step(action)
        obs_flat = obs.astype(np.float32).flatten()
        info["context"] = np.array([self._current_vehicle_id], dtype=np.float32)
        return obs_flat, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._env.render()

    def close(self):
        self._env.close()