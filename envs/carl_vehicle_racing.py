from collections import deque

import gymnasium as gym
import numpy as np
from envs.carl.carl_vehicle_racing import CustomCarRacing, PARKING_GARAGE

class CARLVehicleRacingWrapper(gym.Env):
    """
    Memory-RL compatible wrapper for CARL Vehicle Racing.

    - Stacks the last `frame_stack` frames as CHW channels, oldest first, and
      flattens them: channels [0:3] are t-1 and [3:6] are t for frame_stack=2.
      A single frame quantizes speed coarsely and hides the slip angle entirely.
    - Randomly samples the vehicle type and, when `num_tracks > 0`, one of
      `num_tracks` fixed tracks each episode (the inner env is reseeded with the
      track id). `num_tracks <= 0` regenerates a fresh track every episode (CARL).
    - info["context"] = one-hot(vehicle over PARKING_GARAGE) ++ one-hot(track)
    - Observation space: Box(3*frame_stack*96*96,) float32 [0, 255]; the CNN
      encoder owns the /255 normalization (torchkit/networks.py::ImageEncoder)
    - Action space: Box(3,) float32 [-1, 1]
    """

    FRAME_SHAPE = (3, 96, 96)  # C, H, W of one rendered frame
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, vehicle_ids=None, render_mode=None, frame_skip=1,
                 frame_stack=2, num_tracks=10):
        super().__init__()
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        assert self.frame_skip >= 1, "frame_skip must be >= 1"
        self.frame_stack = frame_stack
        assert self.frame_stack >= 1, "frame_stack must be >= 1"
        self.num_tracks = num_tracks
        # Read by envs/make_env.py -> Learner.init_env to set
        # config_seq.image_encoder.image_shape, so the CNN always matches.
        self.image_shape = (3 * frame_stack, 96, 96)
        if vehicle_ids is None:
            vehicle_ids = [0]  # default: RaceCar only
        self.vehicle_ids = vehicle_ids
        self.vehicle_classes = [PARKING_GARAGE[vid] for vid in vehicle_ids]

        self._env = CustomCarRacing(
            vehicle_class=self.vehicle_classes[0],
            verbose=False,
            render_mode=render_mode,
        )

        # Obs: flattened stacked image (float32 for buffer compatibility)
        self.frame_dim = 96 * 96 * 3  # 27648
        self.obs_dim = self.frame_dim * frame_stack
        self._frames = deque(maxlen=frame_stack)
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
        self._current_track_id = None
        # Vehicle one-hot over the full garage (not the subset, so context_dim is
        # the same for every vehicle_ids subset), followed by the track one-hot.
        self._n_vehicle_ctx = len(PARKING_GARAGE)
        self._context = np.zeros(
            self._n_vehicle_ctx + max(num_tracks, 0), dtype=np.float32
        )

    def _flatten_frame(self, obs):
        """CarRacing renders HWC; ImageEncoder reshapes to image_shape (CHW)."""
        return np.ascontiguousarray(
            obs.transpose(2, 0, 1), dtype=np.float32
        ).ravel()

    def _stacked_obs(self):
        """Oldest frame first; concatenating flat CHW frames gives (3k, 96, 96)."""
        return np.concatenate(self._frames)

    def reset(self, seed=None, options=None, **kwargs):
        # Sample random vehicle
        super().reset(seed=seed)
        idx = int(self.np_random.integers(0, len(self.vehicle_ids)))  # use env's seeded RNG
        self._current_vehicle_id = self.vehicle_ids[idx]
        self._env.vehicle_class = self.vehicle_classes[idx]
        self._context[:] = 0.0
        self._context[self._current_vehicle_id] = 1.0
        if self.num_tracks > 0:
            self._current_track_id = int(self.np_random.integers(0, self.num_tracks))
            self._context[self._n_vehicle_ctx + self._current_track_id] = 1.0
            seed = self._current_track_id  # track id doubles as the generator seed

        obs, info = self._env.reset(seed=seed, options=options)
        frame = self._flatten_frame(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):  # no history yet: repeat the first frame
            self._frames.append(frame)
        info["context"] = self._context.copy()
        return self._stacked_obs(), info

    def step(self, action):
        # Rescale from [-1,1] to each dimension's actual bounds
        action = (action + 1.0) / 2.0 * (self._action_high - self._action_low) + self._action_low
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info = {}
        blown_up = False
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            hull = self._env.car.hull
            if not (np.isfinite(hull.angle)
                    and np.isfinite(hull.position[0])
                    and np.isfinite(hull.position[1])):
                print(f"[CARLVehicleRacing] NaN blowup: vehicle_id={self._current_vehicle_id}, "
                      f"angle={hull.angle}, pos=({hull.position[0]}, {hull.position[1]})")
                total_reward = -100.0
                terminated = True
                blown_up = True
                info["nan_blowup"] = True
                break
            if terminated or truncated:
                break
        # The zero-fill is already CHW, so it bypasses the HWC transpose.
        self._frames.append(
            np.zeros(self.frame_dim, dtype=np.float32)
            if blown_up
            else self._flatten_frame(obs)
        )
        info["context"] = self._context.copy()
        info["success"] = bool(info.get("lap_finished", False))
        return self._stacked_obs(), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        return self._env.render()

    def close(self):
        self._env.close()
