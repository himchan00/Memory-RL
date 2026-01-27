import metaworld
import random
import gymnasium as gym

class MLWrapper(gym.Wrapper):
    def __init__(self, env_name: str, mode: str, render_mode: str=None):
        self.env_name = env_name
        self.mode = mode
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

        # Initialize inner env once to setup the Wrapper
        inner = self._make_inner_env()
        super().__init__(inner)
        

    def _make_inner_env(self, name: str=None):
        if name is None:
            name = random.choice(list(self.classes.keys()))
        env = self.classes[name](render_mode=self._render_mode_cfg, camera_id=1)
        self.name = name
        # Pick a random task for this env
        task = random.choice([t for t in self.tasks if t.env_name == name])
        env.set_task(task)
        # Set max episode steps
        env.max_episode_steps = env.max_path_length 
        return env

    def reset(self, **kwargs):
        # Re-create inner env each episode
        if "name" in kwargs:
            name = kwargs.pop("name")
        else:
            name = None
        self.env = self._make_inner_env(name)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["name"] = self.name
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._render_mode_cfg is None:
            return None
        return self.env.render()

