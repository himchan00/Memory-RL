"""Symbolic Alchemy (DeepMind Alchemy meta-RL benchmark, arXiv:2102.02926).

Gymnasium adapter over dm_alchemy's pure-Python ``symbolic_alchemy`` (no
Docker/Unity/GL). One gym episode = one dm_env episode = ``num_trials`` trials
sharing one hidden "chemistry". With ``end_trial_action=False`` each trial is
exactly ``max_steps_per_trial`` steps, giving a fixed
``num_trials * max_steps_per_trial`` horizon. Run with ``--k 1`` (the multi-trial
structure is native; do NOT use ``KEpisodeWrapper``).

The ground-truth chemistry is exposed as ``info["context"]`` for the oracle
baseline. dm_alchemy is an archived special install
(``scripts/install_dm_alchemy.sh``), hence the lazy import.
"""
import gymnasium as gym
import numpy as np

# see_chemistries key for the ground-truth chemistry (used as oracle context).
_CHEM_KEY = "chem_gt"


class SymbolicAlchemyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level_name, num_trials=10, max_steps_per_trial=20,
                 observe_used=True, add_trial_flag=True, render_mode=None, **_):
        super().__init__()
        self.level_name = level_name
        self.num_trials = int(num_trials)
        self.max_steps_per_trial = int(max_steps_per_trial)
        self.observe_used = bool(observe_used)
        self.add_trial_flag = bool(add_trial_flag)
        self.render_mode = render_mode
        self.max_episode_steps = self.num_trials * self.max_steps_per_trial

        self._seed = None
        self._env = None
        self._t = 0
        self._build_env(seed=None)  # needed to read the specs below

        act_spec = self._env.action_spec()
        self.action_space = gym.spaces.Discrete(int(act_spec.maximum) + 1)
        obs_dim = int(self._env.observation_spec()["symbolic_obs"].shape[0])
        if self.add_trial_flag:
            obs_dim += 1  # soft-reset channel: 1.0 on the first step of each trial
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _build_env(self, seed):
        # Lazy import: the repo only depends on dm_alchemy when this env is used.
        from dm_alchemy import symbolic_alchemy
        from dm_alchemy.types import utils as alchemy_utils
        see = {_CHEM_KEY: alchemy_utils.ChemistrySeen(
            content=alchemy_utils.ElementContent.GROUND_TRUTH)}
        self._env = symbolic_alchemy.get_symbolic_alchemy_level(
            level_name=self.level_name,
            observe_used=self.observe_used,
            end_trial_action=False,  # fixed-length trials -> fixed episode horizon
            num_trials=self.num_trials,
            max_steps_per_trial=self.max_steps_per_trial,
            seed=seed,
            see_chemistries=see,
        )
        self._seed = seed

    def _split_obs(self, ts, trial_flag):
        obs = np.asarray(ts.observation["symbolic_obs"], dtype=np.float32)
        if self.add_trial_flag:
            obs = np.concatenate([obs, np.array([trial_flag], dtype=np.float32)])
        context = np.asarray(ts.observation[_CHEM_KEY], dtype=np.float32)
        return obs, context

    def reset(self, seed=None, options=None):
        # Reseeding rebuilds the dm_env (cheap; once per worker at construction).
        # Unseeded resets advance the RNG -> a new chemistry each episode.
        # ``keep_context`` is ignored: one chemistry per gym episode (run --k 1).
        if seed is not None and seed != self._seed:
            self._build_env(seed=seed)
        ts = self._env.reset()
        self._t = 0
        obs, context = self._split_obs(ts, trial_flag=1.0)
        return obs, {"success": False, "context": context}

    def step(self, action):
        ts = self._env.step(int(action))
        self._t += 1
        reward = float(ts.reward) if ts.reward is not None else 0.0
        truncated = bool(ts.last())
        # dm_alchemy auto-advances to the next trial within the same step; flag
        # the first obs of each new trial (not the terminal step).
        trial_flag = 1.0 if (not truncated and self._t % self.max_steps_per_trial == 0) else 0.0
        obs, context = self._split_obs(ts, trial_flag=trial_flag)
        return obs, reward, False, truncated, {"success": False, "context": context}

    def render(self):
        return None  # symbolic env: nothing to render
