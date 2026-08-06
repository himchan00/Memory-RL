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
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

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
        self._last_action, self._last_reward, self._cum_reward = None, 0.0, 0.0
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
        self._last_action, self._last_reward, self._cum_reward = None, 0.0, 0.0
        obs, context = self._split_obs(ts, trial_flag=1.0)
        return obs, {"success": False, "context": context, "soft_reset": True}

    def step(self, action):
        ts = self._env.step(int(action))
        self._t += 1
        reward = float(ts.reward) if ts.reward is not None else 0.0
        self._last_action, self._last_reward = int(action), reward
        self._cum_reward += reward
        truncated = bool(ts.last())
        # dm_alchemy auto-advances to the next trial within the same step; flag
        # the first obs of each new trial (not the terminal step).
        trial_flag = 1.0 if (not truncated and self._t % self.max_steps_per_trial == 0) else 0.0
        obs, context = self._split_obs(ts, trial_flag=trial_flag)
        return obs, reward, False, truncated, {
            "success": False,
            "context": context,
            "soft_reset": bool(trial_flag),
        }

    def render(self):
        # Compatible with the repo's visualize_env path: returns a fixed-size
        # (H, W, 3) uint8 RGB frame when render_mode="rgb_array", else None.
        if self.render_mode is None:
            return None
        return self._render_frame()

    @staticmethod
    def _decode_action(a):
        if a is None or a == 0:
            return "no-op" if a == 0 else "-"
        stone, tgt = (a - 1) // 13, (a - 1) % 13
        return f"stone{stone} -> " + ("cauldron" if tgt == 0 else f"potion{tgt - 1}")

    def _render_frame(self):
        # Draw the ground-truth chemistry (latent cube + potion graph, reward-
        # colored nodes), the current trial's stones, and a status panel.
        from collections import Counter
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from dm_alchemy.types import stones_and_potions as sp, graphs

        fig = Figure(figsize=(9.0, 4.5), dpi=100)
        canvas = FigureCanvasAgg(fig)
        axg = fig.add_axes([0.02, 0.05, 0.60, 0.88])
        axt = fig.add_axes([0.64, 0.05, 0.35, 0.88]); axt.axis("off")
        axg.set_xticks([]); axg.set_yticks([]); axg.margins(0.15)

        cur_nodes, potions = [], []
        try:  # internals of an archived pkg -> stay defensive
            latent = sp.possible_latent_stones()
            coords = np.array([np.asarray(s.latent_coords, float) for s in latent])
            rewards = np.array([s.reward() for s in latent], float)
            adj = np.array(graphs.convert_graph_to_adj_mat(self._env._chemistry.graph))
            cur_nodes = [n.idx for n in self._env.game_state.existing_stone_nodes()]
            potions = list(self._env.game_state.existing_potions())
            # 2D oblique projection of the cube corners
            px = coords[:, 0] + 0.5 * coords[:, 2]
            py = coords[:, 1] + 0.35 * coords[:, 2]
            for i in range(8):  # faint = a possible (1-bit-flip) edge
                for j in range(i + 1, 8):
                    if int(np.abs(coords[i] - coords[j]).sum()) == 2:
                        axg.plot([px[i], px[j]], [py[i], py[j]], color="0.85", lw=1, zorder=1)
            for i in range(8):  # solid = an edge present in this chemistry
                for j in range(8):
                    if adj[i, j] > 0:
                        axg.plot([px[i], px[j]], [py[i], py[j]], color="0.4", lw=1.6, zorder=2)
            axg.scatter(px, py, c=rewards, cmap="RdYlGn", s=620, edgecolor="k", zorder=3)
            for i in range(8):
                axg.annotate(f"{int(rewards[i]):+d}", (px[i], py[i]), ha="center", va="center", fontsize=8, zorder=4)
            axg.scatter([px[int(np.argmax(rewards))]], [py[int(np.argmax(rewards))]], s=1150,
                        facecolors="none", edgecolors="gold", linewidths=2.5, zorder=3)  # target
            for node in set(cur_nodes):  # current stone(s)
                axg.scatter([px[node]], [py[node]], s=880, facecolors="none",
                            edgecolors="tab:blue", linewidths=3, zorder=5)
        except Exception as ex:  # never let a broken frame kill the eval rollout
            axg.text(0.5, 0.5, f"(chemistry render unavailable)\n{type(ex).__name__}",
                     ha="center", va="center", transform=axg.transAxes)
        axg.set_title("Chemistry (ground truth): latent cube + potion graph", fontsize=9)

        trial = (self._t - 1) // self.max_steps_per_trial if self._t > 0 else 0
        sit = (self._t - 1) % self.max_steps_per_trial if self._t > 0 else 0
        pc = Counter((p.dimension, p.direction) for p in potions)
        pot_items = [f"a{d}{'+' if s > 0 else '-'}x{n}" for (d, s), n in sorted(pc.items())]
        pot_lines = ["  " + "  ".join(pot_items[i:i + 3]) for i in range(0, len(pot_items), 3)] or ["  -"]
        lines = [
            f"level: ...{self.level_name.split('with_')[-1]}",
            f"trial: {trial}/{self.num_trials - 1}   step: {sit}/{self.max_steps_per_trial - 1}",
            f"last action: {self._decode_action(self._last_action)}",
            f"last reward: {self._last_reward:+.1f}",
            f"episode return: {self._cum_reward:+.1f}",
            "",
            f"stones @ nodes: {sorted(cur_nodes)}",
            "potions (axis,dir):", *pot_lines,
            "",
            "gold ring = target (max reward)",
            "blue ring = current stone(s)",
            "gray = possible edge, dark = present",
        ]
        axt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=9, family="monospace")

        canvas.draw()
        return np.ascontiguousarray(np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3])
