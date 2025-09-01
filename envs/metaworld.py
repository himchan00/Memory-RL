import metaworld
import random
class ml_env:
    def __init__(self, env_name: str, mode: str, render_mode: str=None):
        self.env_name = env_name
        self.mode = mode
        self.render_mode = render_mode
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
        self.reset()
        
    def __getattr__(self, name):
        return getattr(self.env, name) # to access gym.Env methods

    def step(self, action):
        return self.env.step(action)
    

    def reset(self, **kwargs):
        name = random.choice(list(self.classes.keys()))
        self.env = self.classes[name](render_mode=self.render_mode) 
        self.task = random.choice([task for task in self.tasks if task.env_name == name])
        self.env.set_task(self.task)
        self.env.max_episode_steps = self.env.max_path_length # 500 for now
        return self.env.reset(**kwargs)

    def render(self):
        return self.env.render()
