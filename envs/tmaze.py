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



class TMazeBase(gym.Env):
    def __init__(
        self,
        episode_length: int = 11,
        corridor_length: int = 10,
        oracle_length: int = 0,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        expose_goal: bool = False,
        add_timestep: bool = False,
    ):
        """
        The Base class of TMaze, decouples episode_length and corridor_length

        Other variants:
            (Osband, 2016): distract_reward = eps > 0, goal_reward is given at T-junction (no choice).
                This only tests the exploration and discounting of agent, no memory required
            (Osband, 2020): ambiguous_position = True, add_timestep = True, supervised = True.
                This only tests the memory of agent, no exploration required (not implemented here)
        """
        super().__init__()
        assert corridor_length >= 1 and episode_length >= 1
        assert penalty <= 0.0

        self.episode_length = episode_length
        self.corridor_length = corridor_length
        self.oracle_length = oracle_length

        self.goal_reward = goal_reward
        self.penalty = penalty
        self.distract_reward = distract_reward

        self.expose_goal = expose_goal
        self.add_timestep = add_timestep

        self.action_space = gym.spaces.Discrete(4)  # four directions
        self.action_mapping = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        self.tmaze_map = np.zeros(
            (3 + 2, self.oracle_length + self.corridor_length + 1 + 2), dtype=bool
        )
        self.bias_x, self.bias_y = 1, 2
        self.tmaze_map[self.bias_y, self.bias_x : -self.bias_x] = True  # corridor
        self.tmaze_map[
            [self.bias_y - 1, self.bias_y + 1], -self.bias_x - 1
        ] = True  # goal candidates
        print(self.tmaze_map.astype(np.int32))

        obs_dim = 3

        if self.add_timestep:
            obs_dim += 1

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def position_encoding(self, x: int, y: int, goal_y: int):
        if x == 0:
            # oracle position
            if not self.oracle_visited:
                # only appear at first
                exposure = goal_y
                self.oracle_visited = True
            else:
                # exposure = 0
                exposure = goal_y # allow multiple exposures for fully observable setting
        else:
            exposure = 0

        if self.expose_goal:
            return [x, y, goal_y if self.oracle_visited else 0]
        x = 2 * x / self.corridor_length - 1  # normalize to [-1, 1]
        return [x, y, exposure]

    def timestep_encoding(self):
        return (
            [
                self.time_step,
            ]
            if self.add_timestep
            else []
        )

    def get_obs(self):
        return np.array(
            self.position_encoding(self.x, self.y, self.goal_y)
            + self.timestep_encoding(),
            dtype=np.float32,
        )

    def reward_fn(self, done: bool, x: int, y: int, goal_y: int):
        if done:  # only give bonus at the final time step
            return float(y == goal_y) * self.goal_reward, bool(y == goal_y)
        else:
            # a penalty (when t > o) if x < t - o (desired: x = t - o)
            rew = float(x < self.time_step - self.oracle_length) * self.penalty
            if x == 0:
                return rew + self.distract_reward, False
            else:
                return rew, False

    def step(self, action):
        self.time_step += 1
        assert self.action_space.contains(action)

        # transition
        move_x, move_y = self.action_mapping[action]
        if self.tmaze_map[self.bias_y + self.y + move_y, self.bias_x + self.x + move_x]:
            # valid move
            self.x, self.y = self.x + move_x, self.y + move_y

        done = self.time_step >= self.episode_length
        rew, success = self.reward_fn(done, self.x, self.y, self.goal_y)
        return self.get_obs(), rew, success, done, {"success": success}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        self.x, self.y = self.oracle_length, 0
        self.goal_y = self.np_random.choice([-1, 1])
        self.oracle_visited = False
        self.time_step = 0
        return self.get_obs(), {"success": False}

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


class TMazeClassicPassive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        **kwargs
    ):
        """
        Classic TMaze with Passive Memory
            assert episode_length == corridor_length + 1
            (Bakker, 2001): ambiguous_position = True. penalty = 0
                This is too hard even for T = 10 for vanilla agents because the exploration is extremely hard.
                This tests both memory and exploration
            **(tmaze_classic; this work)**: based on (Bakker, 2001), set penalty < 0
                Unit-tests memory
        """
        super().__init__(
            episode_length=corridor_length + 1,
            corridor_length=corridor_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            add_timestep=False,
        )


class TMazeClassicActive(TMazeBase):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        penalty: float = 0.0,
        distract_reward: float = 0.0,
        **kwargs
    ):
        """
        Classic TMaze with Active Memory
            assert episode_length == corridor_length + 1 + 2o
            where o is the length between the starting point and oracle that gives the goal information
            TMazeClassicPassive is a special case of o = 0.
        """
        oracle_length = 1
        super().__init__(
            episode_length=corridor_length + 2 * oracle_length + 1,
            corridor_length=corridor_length,
            oracle_length=oracle_length,
            goal_reward=goal_reward,
            penalty=penalty,
            distract_reward=distract_reward,
            expose_goal=False,
            add_timestep=False,
        )


class TMazeDetour(gym.Env):
    def __init__(
        self,
        corridor_length: int = 10,
        goal_reward: float = 1.0,
        distractors: bool = False,
        add_timestep: bool = False,
        **kwargs
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


