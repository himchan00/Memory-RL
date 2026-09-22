import metaworld
import random
import numpy as np
import gymnasium as gym


_BENCHMARK_SPECS = {}


def _compute_rand_vec_dim(benchmark) -> int:
    """Probe each train class once to find the maximum rand_vec dimension."""
    max_dim = 0
    for name in sorted(benchmark.train_classes.keys()):
        env = benchmark.train_classes[name]()
        task = next(t for t in benchmark.train_tasks if t.env_name == name)
        env.set_task(task)
        rand_vec = getattr(env.unwrapped, "_last_rand_vec", None)
        if rand_vec is None:
            rand_vec = env.unwrapped._random_reset_space.low
        max_dim = max(max_dim, int(np.asarray(rand_vec).shape[0]))
        env.close()
    return max_dim


def get_benchmark_spec(env_name: str):
    """Build (benchmark, rand_vec_dim) for `env_name` once per process.

    `metaworld.ML45()` instantiates every task env to sample its tasks and the
    rand_vec probe instantiates all 45 train classes again. Together that costs
    ~40 s and ~450 MB of *retained* RSS (MuJoCo model memory is not returned to
    the OS), although the resulting benchmark pickles to under 1 MB. Building it
    inside each `AsyncVectorEnv` worker therefore OOM-kills the run on a 64 GB
    box with the default `n_env=64` (128 workers x 450 MB). `main.py` calls this
    in the parent and hands the spec to the workers instead.
    """
    if env_name not in _BENCHMARK_SPECS:
        if env_name == "ML10":
            benchmark = metaworld.ML10()
        elif env_name == "ML45":
            benchmark = metaworld.ML45()
        else:
            raise ValueError(f"Unknown environment name: {env_name}")
        _BENCHMARK_SPECS[env_name] = (benchmark, _compute_rand_vec_dim(benchmark))
    return _BENCHMARK_SPECS[env_name]


class MLWrapper(gym.Wrapper):
    def __init__(self, env_name: str, mode: str, render_mode: str=None, max_episode_steps: int=None,
                 benchmark_spec=None):
        self.env_name = env_name
        self.mode = mode
        self._max_episode_steps_override = max_episode_steps
        # Store desired render mode under a different name to avoid clashing
        self._render_mode_cfg = render_mode
        if benchmark_spec is None:
            benchmark_spec = get_benchmark_spec(env_name)
        self.benchmark, self._rand_vec_dim = benchmark_spec

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
        self._context_dim = self._n_classes + self._rand_vec_dim
        self._cached_context = None

        # Initialize inner env once to setup the Wrapper
        inner = self._make_inner_env()
        super().__init__(inner)
        self.max_episode_steps = int(inner.max_episode_steps)

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

    def _make_inner_env(self, name: str=None, task=None):
        if name is None:
            name = random.choice(list(self.classes.keys()))
        env = self.classes[name](render_mode=self._render_mode_cfg, camera_id=1)
        self.name = name
        # Pick a random task for this env (or reuse the one passed in).
        if task is None:
            task = random.choice([t for t in self.tasks if t.env_name == name])
        self._current_task = task
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
        options = kwargs.get("options") or {}
        keep_context = bool(options.get("keep_context", False))
        if "name" in kwargs:
            name = kwargs.pop("name")
            task = None
        elif keep_context and getattr(self, "name", None) is not None:
            # soft reset: preserve the same class + task across resets.
            name = self.name
            task = getattr(self, "_current_task", None)
        else:
            name = None
            task = None
        kwargs.pop("options", None)  # don't forward our custom option downstream
        self.env = self._make_inner_env(name, task)
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
