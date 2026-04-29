import metaworld
import random
import numpy as np
import gymnasium as gym

class MLWrapper(gym.Wrapper):
    def __init__(self, env_name: str, mode: str, render_mode: str=None, max_episode_steps: int=None):
        self.env_name = env_name
        self.mode = mode
        self._max_episode_steps_override = max_episode_steps
        # Store desired render mode under a different name to avoid clashing
        self._render_mode_cfg = render_mode
        if env_name == "ML10":
            self.benchmark = metaworld.ML10()
        elif env_name == "ML45":
            self.benchmark = metaworld.ML45()
        else:
            raise ValueError(f"Unknown environment name: {env_name}")

        if mode == "train":
            self.classes = self.benchmark.train_classes
            self.tasks = self.benchmark.train_tasks
        elif mode == "test":
            self.classes = self.benchmark.test_classes
            self.tasks = self.benchmark.test_tasks
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Build oracle context spec from train_classes only (test split unused).
        self._class_names = sorted(self.benchmark.train_classes.keys())
        self._class_to_idx = {n: i for i, n in enumerate(self._class_names)}
        self._n_classes = len(self._class_names)
        self._rand_vec_dim = self._compute_rand_vec_dim()
        self._context_dim = self._n_classes + self._rand_vec_dim
        self._cached_context = None

        # Initialize inner env once to setup the Wrapper
        inner = self._make_inner_env()
        super().__init__(inner)

    def _compute_rand_vec_dim(self) -> int:
        """Probe each train class once to find the maximum rand_vec dimension."""
        max_dim = 0
        for name in self._class_names:
            env = self.benchmark.train_classes[name]()
            task = next(t for t in self.benchmark.train_tasks if t.env_name == name)
            env.set_task(task)
            rand_vec = getattr(env.unwrapped, "_last_rand_vec", None)
            if rand_vec is None:
                rand_vec = env.unwrapped._random_reset_space.low
            max_dim = max(max_dim, int(np.asarray(rand_vec).shape[0]))
            env.close()
        return max_dim

    def _build_context(self, name: str, env: gym.Env) -> np.ndarray:
        one_hot = np.zeros(self._n_classes, dtype=np.float32)
        one_hot[self._class_to_idx[name]] = 1.0
        rand_vec = np.asarray(
            getattr(env.unwrapped, "_last_rand_vec", np.zeros(self._rand_vec_dim)),
            dtype=np.float32,
        )
        padded = np.zeros(self._rand_vec_dim, dtype=np.float32)
        padded[: rand_vec.shape[0]] = rand_vec
        return np.concatenate([one_hot, padded], axis=-1)

    def _make_inner_env(self, name: str=None):
        if name is None:
            name = random.choice(list(self.classes.keys()))
        env = self.classes[name](render_mode=self._render_mode_cfg, camera_id=1)
        self.name = name
        # Pick a random task for this env
        task = random.choice([t for t in self.tasks if t.env_name == name])
        env.set_task(task)
        # Set max episode steps
        if self._max_episode_steps_override:
            env.max_path_length = self._max_episode_steps_override
        env.max_episode_steps = env.max_path_length
        # Cache oracle context for the lifetime of this episode (task is fixed).
        self._cached_context = self._build_context(name, env)
        return env

    def reset(self, **kwargs):
        # Re-create inner env each episode
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = None
        self.env = self._make_inner_env(name)
        obs, info = self.env.reset(**kwargs)
        info["name"] = self.name
        info["context"] = self._cached_context.copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["name"] = self.name
        info["context"] = self._cached_context.copy()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._render_mode_cfg is None:
            return None
        return self.env.render()
