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

    def __init__(self, vehicle_ids=None, render_mode=None, frame_skip=1):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        assert self.frame_skip >= 1, "frame_skip must be >= 1"
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
        self._current_vehicle_id = None

    def reset(self, seed=None, options=None, **kwargs):
        # Sample random vehicle
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self.vehicle_ids)))  # use env's seeded RNG
        #idx = np.random.randint(len(self.vehicle_ids))
        self._current_vehicle_id = self.vehicle_ids[idx]
        self._env.vehicle_class = self.vehicle_classes[idx]

        obs, info = self._env.reset(seed=seed, options=options)
        obs_flat = obs.astype(np.float32).flatten()  # (27648,)
        info["context"] = np.array([self._current_vehicle_id], dtype=np.float32)
        return obs_flat, info

    def step(self, action):
        # Rescale from [-1,1] to each dimension's actual bounds
        action = (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            hull = self._env.car.hull
            if not (np.isfinite(hull.angle)
                    and np.isfinite(hull.position[0])
                    and np.isfinite(hull.position[1])):
                print(f"[CARLVehicleRacing] NaN blowup: vehicle_id={self._current_vehicle_id}, "
                      f"angle={hull.angle}, pos=({hull.position[0]}, {hull.position[1]})")
                obs = np.zeros(self.IMAGE_SHAPE, dtype=np.uint8)
                total_reward = -100.0
                terminated = True
                info["nan_blowup"] = True
                break
            if terminated or truncated:
                break
        obs_flat = obs.astype(np.float32).flatten()
        info["context"] = np.array([self._current_vehicle_id], dtype=np.float32)
        info["success"] = bool(info.get("lap_finished", False))
        return obs_flat, total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._env.render()

    def close(self):
        self._env.close()