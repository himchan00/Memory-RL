import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
from gymnasium.utils import seeding

"""
T-Maze: originated from (Bakker, 2001) and earlier neuroscience work, 
    and here extended to unit-test several key challenges in RL:
- Exploration
- Memory and credit assignment
- Discounting and distraction
- Generalization

Finite horizon problem: episode_length
Has a corridor of corridor_length
Looks like
                        g1
o--s---------------------j
                        g2
o is the oracle point, (x, y) = (0, 0)
s is starting point, (x, y) = (o, 0)
j is T-juncation, (x, y) = (o + corridor_length, 0)
g1 is goal candidate, (x, y) = (o + corridor_length, 1)
g2 is goal candidate, (x, y) = (o + corridor_length, -1)
"""





class TMazeDetour(gym.Env):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        distractors: bool = False,
        add_timestep: bool = False,
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1
        self.corridor_length = corridor_length
        if distractors == True:
            self.episode_length = self.corridor_length + 3 * 3 + 1 # 3 detours
        else:
            self.episode_length = self.corridor_length + 3 + 1 # 1 detours
        self.goal_reward = goal_reward
        self.penalty = - 1 / self.episode_length
        self.x_dt = self.corridor_length // 2 # Detour position
        self.distractors = distractors
        self.x_d1, self.x_d2 = self.corridor_length // 4, self.corridor_length * 3 // 4 # distractor positions. only used when distractors=True
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)  # four directions
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[
            [self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1
        ] = True  # goal candidates
        print(self.tmaze_map.astype(np.int32))

        obs_dim = 2
        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )


    def get_obs(self):
        position = [2 * self.x / self.corridor_length - 1, self.y]  # normalize to [-1, 1]
        time = [2 * self.time_step / self.episode_length - 1] if self.add_timestep else []
        return np.array(position + time, dtype=np.float32)

            
    def reward_fn(self, success: bool, x: int, x_prev: int, y: int, y_prev: int):
        if success: 
            return self.goal_reward
        else:
            delta_x, delta_y = x - x_prev, y - y_prev
            rew = self.penalty
            if abs(delta_x) > 0:
                rew -= delta_x * self.penalty # + x velocity reward (Note: self.penalty is negative)
            if abs(delta_y) > 0: 
                if self.x == (self.x_dt - 1) or self.x == (self.x_d1 - 1) or self.x == (self.x_d2 - 1):
                    rew -= (abs(y) - abs(y_prev)) * self.penalty # When the agent meets detour or distractors, it must go up or down
                elif self.x == (self.x_dt + 1) or self.x == (self.x_d1 + 1) or self.x == (self.x_d2 + 1):
                    rew -= (abs(y_prev) - abs(y)) * self.penalty # When the agent passes detour or distractors, it must go to the middle
                else:
                    ValueError("Unreachable code reached")
            return rew

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        x_prev, y_prev = self.x, self.y
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = x_prev + move_x, y_prev + move_y

        truncated = False
        success = False
        info = {}
        if (self.x, self.y) == (self.corridor_length, self.goal_y):
            # reached the goal
            success = True
            info["success"] = True
        if self.time_step >= self.episode_length:
            truncated = True

        rew = self.reward_fn(success, self.x, x_prev, self.y, y_prev)

        return self.get_obs(), rew, success, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        self.x, self.y = 0, 0
        self.goal_y = self.np_random.choice([-1, 1])
        self.place_detours()
        self.time_step = 0
        return self.get_obs(), {}

    def place_detours(self):
        """
        Place detours in the T-Maze.
        The detour are placed at corridor_length // 2
        If distractors == True, distractor detours are placed at corridor_length * 1 // 4 & corridor_length * 3 // 4
        """
        # Block the corridor at the detour positions
        self.tmaze_map[self.bias_y, self.bias_x + self.x_dt] = False
        if self.distractors:
            self.x_d1 = self.corridor_length // 4
            self.x_d2 = self.corridor_length * 3 // 4
            self.tmaze_map[self.bias_y, self.bias_x + self.x_d1] = False
            self.tmaze_map[self.bias_y, self.bias_x + self.x_d2] = False

        # Place detours
        self.tmaze_map[self.bias_y + self.goal_y, self.bias_x + self.x_dt - 1 : self.bias_x + self.x_dt + 2] = True
        self.tmaze_map[self.bias_y - self.goal_y, self.bias_x + self.x_dt - 1 : self.bias_x + self.x_dt + 2] = False
        if self.distractors:
            y_dis = self.np_random.choice([-1, 1], size=2, replace=True) # for distractor detours
            self.tmaze_map[self.bias_y + y_dis[0], self.bias_x + self.x_d1 - 1 : self.bias_x + self.x_d1 + 2] = True
            self.tmaze_map[self.bias_y - y_dis[0], self.bias_x + self.x_d1 - 1 : self.bias_x + self.x_d1 + 2] = False
            self.tmaze_map[self.bias_y + y_dis[1], self.bias_x + self.x_d2 - 1 : self.bias_x + self.x_d2 + 2] = True
            self.tmaze_map[self.bias_y - y_dis[1], self.bias_x + self.x_d2 - 1 : self.bias_x + self.x_d2 + 2] = False

        return self.tmaze_map
    
    def visualize(self, trajectories: np.array, idx: str):
        from utils import logger

        # trajectories: (B, T+1, O)
        batch_size, seq_length, _ = trajectories.shape
        xs = np.arange(seq_length)

        for traj in trajectories:
            # plot the 0-th element
            plt.plot(xs, traj[:, 0])

        plt.xlabel("Time Step")
        plt.ylabel("Position X")
        plt.savefig(
            os.path.join(logger.get_dir(), "plt", f"{idx}.png"),
            dpi=200,  # 200
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close()


