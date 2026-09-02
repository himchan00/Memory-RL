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
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

# see_chemistries key for the ground-truth chemistry (used as oracle context).
_CHEM_KEY = "chem_gt"
NO_OP_ACTION = 0
ALCHEMY_ACTION_CATEGORY_NAMES = ("no_op", "cash", "potion")
ALCHEMY_ACTION_CATEGORY_NO_OP = 0
ALCHEMY_ACTION_CATEGORY_CASH = 1
ALCHEMY_ACTION_CATEGORY_POTION = 2


@dataclass(frozen=True)
class DecodedAlchemyAction:
    kind: str
    stone_index: int | None = None
    potion_index: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "kind": self.kind,
            "stone_index": self.stone_index,
            "potion_index": self.potion_index,
        }


@dataclass(frozen=True)
class AlchemyObservationLayout:
    max_stones: int
    max_potions: int
    stone_feature_dim: int
    potion_feature_dim: int
    stone_absent_value: float = 2.0
    potion_absent_value: float = 1.0

    @property
    def potions_per_stone(self) -> int:
        return self.max_potions + 1

    @property
    def symbolic_obs_dim(self) -> int:
        return (
            self.max_stones * self.stone_feature_dim
            + self.max_potions * self.potion_feature_dim
        )


def get_symbolic_alchemy_layout(
    observe_used: bool = True,
    structured_potions: bool = False,
) -> AlchemyObservationLayout:
    from dm_alchemy.symbolic_alchemy import (
        MAX_POTIONS,
        MAX_STONES,
        slot_based_num_features,
    )

    stone_feature_dim, potion_feature_dim = slot_based_num_features(observe_used)
    if structured_potions:
        # axis one-hot (3) + direction (1) [+ used], replacing the ordinal
        # `type_value` scalar. See SymbolicAlchemyEnv._restructure_potions.
        if not observe_used:
            # Without a used flag, absence is read off feature 0, which for a
            # one-hot axis is 0 for both "absent" and "axis != 0".
            raise ValueError("structured_potions requires observe_used=True.")
        potion_feature_dim = 4 + int(observe_used)
    return AlchemyObservationLayout(
        max_stones=int(MAX_STONES),
        max_potions=int(MAX_POTIONS),
        stone_feature_dim=int(stone_feature_dim),
        potion_feature_dim=int(potion_feature_dim),
    )


def encode_cash_action(stone_index: int, *, observe_used: bool = True) -> int:
    layout = get_symbolic_alchemy_layout(observe_used)
    return int(stone_index) * layout.potions_per_stone + 1


def encode_potion_action(
    stone_index: int,
    potion_index: int,
    *,
    observe_used: bool = True,
) -> int:
    layout = get_symbolic_alchemy_layout(observe_used)
    return int(stone_index) * layout.potions_per_stone + int(potion_index) + 2


def decode_action(
    action: int,
    *,
    observe_used: bool = True,
) -> DecodedAlchemyAction:
    if action == NO_OP_ACTION:
        return DecodedAlchemyAction(kind="no_op")
    if action < 0:
        raise ValueError(f"Action must be non-negative, got {action}.")
    layout = get_symbolic_alchemy_layout(observe_used)
    stone_index, target = divmod(action - 1, layout.potions_per_stone)
    if target == 0:
        return DecodedAlchemyAction(kind="cash", stone_index=stone_index)
    return DecodedAlchemyAction(
        kind="potion",
        stone_index=stone_index,
        potion_index=target - 1,
    )


def action_category_ids(
    action: torch.Tensor | np.ndarray | int,
    *,
    observe_used: bool = True,
) -> torch.Tensor | np.ndarray | int:
    layout = get_symbolic_alchemy_layout(observe_used)
    potions_per_stone = layout.potions_per_stone

    if torch.is_tensor(action):
        action = action.to(dtype=torch.long)
        target = torch.remainder(action - 1, potions_per_stone)
        return torch.where(
            action == NO_OP_ACTION,
            torch.full_like(action, ALCHEMY_ACTION_CATEGORY_NO_OP),
            ALCHEMY_ACTION_CATEGORY_CASH + (target > 0).long(),
        )

    action_array = np.asarray(action, dtype=np.int64)
    category = np.full_like(action_array, ALCHEMY_ACTION_CATEGORY_NO_OP)
    positive = action_array > NO_OP_ACTION
    if np.any(positive):
        target = np.remainder(action_array[positive] - 1, potions_per_stone)
        category[positive] = (
            ALCHEMY_ACTION_CATEGORY_CASH + (target > 0).astype(np.int64)
        )
    if np.isscalar(action):
        return int(category.item())
    return category


# Extra scalars appended by SymbolicAlchemyEnv when add_trial_phase=True:
# (steps left in this trial, trials left in this episode), both normalized.
TRIAL_PHASE_DIM = 2

# Extra scalars appended by SymbolicAlchemyEnv when aux_canon_target=True:
# the canonical-frame (latent) description of the current state, laid out as
#   [0:9]   3 stones x 3 latent coordinates, each in {-1, +1}
#   [9:21]  12 potions, each a latent type index in [0, 6)
# Absent / used slots carry AUX_CANON_ABSENT so the consumer masks them out
# rather than reading a magic in-range value.
#
# This is a SUPERVISION TARGET, never a network input: the agent strips these
# dims before anything (RNN_head, critic, action mask) sees the observation.
AUX_CANON_DIM = 21
AUX_CANON_STONE_DIM = 9
AUX_CANON_POTION_DIM = 12
AUX_CANON_NUM_POTION_TYPES = 6
AUX_CANON_ABSENT = -99.0


def _split_symbolic_observation(
    observation: torch.Tensor | np.ndarray,
    *,
    observe_used: bool,
    add_trial_flag: bool,
    context_dim: int,
    structured_potions: bool = False,
    add_trial_phase: bool = False,
    aux_canon_target: bool = False,
) -> tuple[torch.Tensor | np.ndarray, AlchemyObservationLayout]:
    layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
    tail = (
        int(add_trial_flag)
        + (TRIAL_PHASE_DIM if add_trial_phase else 0)
        + (AUX_CANON_DIM if aux_canon_target else 0)
    )
    raw_dim = layout.symbolic_obs_dim + tail
    expected_dim = raw_dim + int(context_dim)
    if observation.shape[-1] != expected_dim:
        raise ValueError(
            "Alchemy observation has unexpected width "
            f"{observation.shape[-1]}; expected {expected_dim}."
        )

    symbolic_obs = observation[..., :raw_dim]
    if tail:
        symbolic_obs = symbolic_obs[..., :-tail]
    return symbolic_obs, layout


def _present_flags(symbolic_obs, layout, observe_used):
    """(stone_present, potion_present) from the SYMBOLIC block only.

    Single source of truth for "is this slot occupied", shared by the action
    mask and by the aux-target loss mask so the two can never disagree.
    """
    stone_width = layout.max_stones * layout.stone_feature_dim
    stone_features = symbolic_obs[..., :stone_width].reshape(
        *symbolic_obs.shape[:-1],
        layout.max_stones,
        layout.stone_feature_dim,
    )
    potion_features = symbolic_obs[..., stone_width:].reshape(
        *symbolic_obs.shape[:-1],
        layout.max_potions,
        layout.potion_feature_dim,
    )
    if observe_used:
        return stone_features[..., -1] < 0.5, potion_features[..., -1] < 0.5
    if torch.is_tensor(symbolic_obs):
        stone_present = torch.any(
            stone_features < (layout.stone_absent_value - 0.5),
            dim=-1,
        )
    else:
        stone_present = np.any(
            stone_features < (layout.stone_absent_value - 0.5),
            axis=-1,
        )
    potion_present = (
        potion_features[..., 0] < layout.potion_absent_value - 1e-6
    )
    return stone_present, potion_present


def present_flags_from_observation(
    observation: torch.Tensor | np.ndarray,
    *,
    observe_used: bool,
    add_trial_flag: bool,
    context_dim: int = 0,
    structured_potions: bool = False,
    add_trial_phase: bool = False,
    aux_canon_target: bool = False,
):
    """Public wrapper: (stone_present, potion_present) with width checking."""
    symbolic_obs, layout = _split_symbolic_observation(
        observation,
        observe_used=observe_used,
        add_trial_flag=add_trial_flag,
        context_dim=context_dim,
        structured_potions=structured_potions,
        add_trial_phase=add_trial_phase,
        aux_canon_target=aux_canon_target,
    )
    return _present_flags(symbolic_obs, layout, observe_used)


def valid_action_mask_from_observation(
    observation: torch.Tensor | np.ndarray,
    *,
    observe_used: bool,
    add_trial_flag: bool,
    context_dim: int = 0,
    structured_potions: bool = False,
    add_trial_phase: bool = False,
    aux_canon_target: bool = False,
) -> torch.Tensor | np.ndarray:
    symbolic_obs, layout = _split_symbolic_observation(
        observation,
        observe_used=observe_used,
        add_trial_flag=add_trial_flag,
        context_dim=context_dim,
        structured_potions=structured_potions,
        add_trial_phase=add_trial_phase,
        aux_canon_target=aux_canon_target,
    )

    stone_present, potion_present = _present_flags(
        symbolic_obs, layout, observe_used
    )

    if torch.is_tensor(symbolic_obs):
        block_valid = torch.cat(
            (
                stone_present.unsqueeze(-1),
                stone_present.unsqueeze(-1) & potion_present.unsqueeze(-2),
            ),
            dim=-1,
        ).reshape(*stone_present.shape[:-1], -1)
        no_op = torch.ones(
            (*stone_present.shape[:-1], 1),
            dtype=torch.bool,
            device=symbolic_obs.device,
        )
        return torch.cat((no_op, block_valid), dim=-1)

    block_valid = np.concatenate(
        (
            stone_present[..., None],
            stone_present[..., None] & potion_present[..., None, :],
        ),
        axis=-1,
    ).reshape(*stone_present.shape[:-1], -1)
    no_op = np.ones((*stone_present.shape[:-1], 1), dtype=np.bool_)
    return np.concatenate((no_op, block_valid), axis=-1)


class SymbolicAlchemyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, level_name, num_trials=10, max_steps_per_trial=20,
                 observe_used=True, add_trial_flag=True, canonicalize_oracle=False,
                 structured_potions=False, structured_stones=False,
                 add_trial_phase=False, aux_canon_target=False,
                 context_graph_only=False, render_mode=None, **_):
        super().__init__()
        self.level_name = level_name
        self.num_trials = int(num_trials)
        self.max_steps_per_trial = int(max_steps_per_trial)
        self.observe_used = bool(observe_used)
        self.add_trial_flag = bool(add_trial_flag)
        self.canonicalize_oracle = bool(canonicalize_oracle)
        self.structured_potions = bool(structured_potions)
        self.structured_stones = bool(structured_stones)
        if self.structured_stones and not self.observe_used:
            # Without a used flag, absence is read off the 2.0 sentinel itself
            # (see valid_action_mask_from_observation); zeroing it would make
            # every slot look present.
            raise ValueError("structured_stones requires observe_used=True.")
        self.add_trial_phase = bool(add_trial_phase)
        self.aux_canon_target = bool(aux_canon_target)
        if self.aux_canon_target and self.canonicalize_oracle:
            # The target IS the canonicalization; with it already applied the
            # supervised map is the identity and teaches nothing.
            raise ValueError(
                "aux_canon_target is the canonical-frame supervision target; "
                "it is meaningless with canonicalize_oracle=True (the "
                "observation is already in the latent frame)."
            )
        self.context_graph_only = bool(context_graph_only)
        if self.context_graph_only and not self.canonicalize_oracle:
            # dims 12-27 are exactly the frame maps; they are redundant only
            # once the frame has already been undone.
            raise ValueError(
                "context_graph_only drops the frame maps from chem_gt, which "
                "the agent still needs unless canonicalize_oracle=True."
            )
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
        if self.structured_potions:
            obs_dim = get_symbolic_alchemy_layout(
                self.observe_used, structured_potions=True).symbolic_obs_dim
        if self.add_trial_flag:
            obs_dim += 1  # soft-reset channel: 1.0 on the first step of each trial
        if self.add_trial_phase:
            obs_dim += TRIAL_PHASE_DIM
        if self.aux_canon_target:
            _layout = get_symbolic_alchemy_layout(self.observe_used)
            if (_layout.max_stones * 3 != AUX_CANON_STONE_DIM
                    or _layout.max_potions != AUX_CANON_POTION_DIM):
                raise ValueError(
                    "AUX_CANON_DIM assumes 3 stones x 3 coords + 12 potions; "
                    f"this level has {_layout.max_stones} stones and "
                    f"{_layout.max_potions} potions."
                )
            obs_dim += AUX_CANON_DIM
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

    def _canonicalize(self, obs):
        """Undo the perceptual frame in place: perceived -> latent coordinates.

        PRIVILEGED. This reads the hidden chemistry, so it is a diagnostic for
        the oracle only -- never enable it for a memory model being evaluated.

        Only two fields change. Each stone's coordinate triple becomes its
        LATENT triple, and each potion's type scalar becomes its LATENT
        (axis, direction) type under the same ``index / 3 - 1`` encoding the env
        uses for the perceived type. Rewards, used flags, absent-slot sentinels
        and the trial flag are left alone, so the observation width and the
        action mask (which reads only the used flags) are unchanged. Absent
        slots keep their sentinel because only *existing* stones/potions are
        iterated.

        This is the transform ``scripts/bc_diagnostic.py:latent_obs`` measured
        as the ``latent`` condition, ported verbatim so the RL result is
        comparable to that BC number.
        """
        layout = get_symbolic_alchemy_layout(self.observe_used)
        stone_width = layout.max_stones * layout.stone_feature_dim
        state = self._env.game_state
        for stone in state.existing_stones():
            slot = state.get_stone_ind(stone_inst=stone.idx)
            base = layout.stone_feature_dim * slot
            obs[base:base + 3] = np.asarray(stone.latent, dtype=np.float32)
        for potion in state.existing_potions():
            slot = state.get_potion_ind(potion_inst=potion.idx)
            latent_type = int(potion.dimension) * 2 + (1 if potion.direction > 0 else 0)
            obs[stone_width + layout.potion_feature_dim * slot] = latent_type / 3.0 - 1.0
        return obs

    def _restructure_potions(self, obs):
        """Ordinal `type_value` scalar -> axis one-hot(3) + direction(1).

        The env packs a potion's type into one scalar, ``index / 3 - 1`` with
        ``index = axis * 2 + (direction > 0)``. That ordering is an artifact:
        it puts the two directions of an axis next to each other on a line the
        network then has to carve into six categories, and it makes "same axis"
        and "same direction" both non-linear in the input.

        This reads that scalar back (whichever frame wrote it, so it composes
        with canonicalization) and re-emits axis and direction as separate
        fields. Absent/used slots decode to index 6 -- the ``1.0`` sentinel --
        and become all-zero with the used flag set, matching the convention
        ``valid_action_mask_from_observation`` reads.
        """
        src = get_symbolic_alchemy_layout(self.observe_used)
        dst = get_symbolic_alchemy_layout(self.observe_used, structured_potions=True)
        stone_width = src.max_stones * src.stone_feature_dim
        potions = obs[stone_width:].reshape(src.max_potions, src.potion_feature_dim)

        out = np.zeros((dst.max_potions, dst.potion_feature_dim), dtype=np.float32)
        for slot in range(src.max_potions):
            index = int(round((float(potions[slot, 0]) + 1.0) * 3.0))
            if not 0 <= index < 6:  # absent or used -> leave zeros, flag it
                out[slot, -1] = 1.0
                continue
            out[slot, index // 2] = 1.0                   # axis one-hot
            out[slot, 3] = 1.0 if index % 2 else -1.0     # direction
            if self.observe_used:
                out[slot, -1] = float(potions[slot, -1])
        return np.concatenate([obs[:stone_width], out.reshape(-1)])

    def _restructure_stones(self, obs):
        """Absent stone: 2.0 sentinel in every field -> all-zero + used flag.

        A stone slot is ``[c0, c1, c2, reward/3, used]``. When the slot is empty
        the env writes the ``stone_absent_value`` sentinel (2.0) into the four
        leading fields -- so the coordinate channels, which otherwise carry
        -1/0/+1, and the reward channel, which otherwise carries a value in
        [-1, 1], both take a magic out-of-range value. The network has to learn
        "2 in this channel is not a coordinate" separately for each field.

        ``_restructure_potions`` already fixed exactly this pathology on the
        potion block (absent -> all-zero, absence signalled solely by the used
        flag). Stones never got the same treatment; this applies it.

        The width is unchanged and ``used`` stays the last feature, so the
        layout, the observation space and
        ``valid_action_mask_from_observation`` all keep working untouched --
        the mask reads ``stone_features[..., -1] < 0.5``, which this preserves.

        NON-PRIVILEGED: reads only the agent's own observation, never the
        hidden chemistry. Safe to enable for a memory model.
        """
        layout = get_symbolic_alchemy_layout(self.observe_used)
        stone_width = layout.max_stones * layout.stone_feature_dim
        stones = obs[:stone_width].reshape(
            layout.max_stones, layout.stone_feature_dim).copy()
        absent = stones[:, -1] >= 0.5          # used flag == absent, as the mask reads it
        stones[absent, :-1] = 0.0              # drop the 2.0 sentinel, keep the flag
        return np.concatenate([stones.reshape(-1), obs[stone_width:]])

    def _aux_canon_targets(self):
        """Canonical-frame (latent) description of the current state.

        ``[3 stones x 3 latent coords] ++ [12 potion latent type indices]``,
        with ``AUX_CANON_ABSENT`` in every slot that holds no stone/potion.
        Read from exactly the same game-state fields ``_canonicalize`` reads,
        so the two agree by construction.

        This is a SUPERVISION TARGET appended to the observation, not an input.
        For the ORACLE configuration it adds NO information: it is a
        deterministic function of the agent's own input (the perceived
        observation plus the ``chem_gt`` frame maps in dims 12-27), a fact
        verified to 100% test accuracy by ``scripts/probe_frame_map.py``. It
        exists purely to give the shared trunk a dense training signal for a
        function the scalar TD signal never drives it to compute.

        (For a MEMORY model with no ``chem_gt`` in the observation the same
        target WOULD be privileged. That is a separate question; do not
        conflate the two.)
        """
        out = np.full(AUX_CANON_DIM, AUX_CANON_ABSENT, dtype=np.float32)
        state = self._env.game_state
        for stone in state.existing_stones():
            slot = state.get_stone_ind(stone_inst=stone.idx)
            out[3 * slot:3 * slot + 3] = np.asarray(stone.latent, dtype=np.float32)
        for potion in state.existing_potions():
            slot = state.get_potion_ind(potion_inst=potion.idx)
            latent_type = int(potion.dimension) * 2 + (1 if potion.direction > 0 else 0)
            out[AUX_CANON_STONE_DIM + slot] = float(latent_type)
        return out

    def _trial_phase(self):
        """(steps left in this trial, trials left in this episode), normalized.

        ``add_trial_flag`` fires a single 1.0 spike on the first step of a trial,
        and ``config_seq.use_pe`` supplies the ABSOLUTE step index in the
        200-step meta-episode. Neither answers the question the cash-in decision
        actually asks -- "how many steps do I have left with these stones before
        the trial resets and I lose them?" -- without the agent first learning
        modular arithmetic on the absolute index.

        Both values are in [0, 1] and monotonically decrease within their unit.
        Recomputed from ``self._t``, which is a property of the wrapper's own
        clock, so this is NON-PRIVILEGED: it reveals nothing about the hidden
        chemistry, only about the schedule the agent is already subject to.
        """
        within = self._t % self.max_steps_per_trial
        steps_left = (self.max_steps_per_trial - within) / self.max_steps_per_trial
        trials_left = (
            self.num_trials - self._t // self.max_steps_per_trial
        ) / self.num_trials
        return np.array([steps_left, max(trials_left, 0.0)], dtype=np.float32)

    def _split_obs(self, ts, trial_flag):
        obs = np.asarray(ts.observation["symbolic_obs"], dtype=np.float32)
        if self.canonicalize_oracle:
            obs = self._canonicalize(np.array(obs, copy=True))
        if self.structured_stones:
            obs = self._restructure_stones(obs)
        if self.structured_potions:
            obs = self._restructure_potions(obs)
        if self.add_trial_flag:
            obs = np.concatenate([obs, np.array([trial_flag], dtype=np.float32)])
        if self.add_trial_phase:
            obs = np.concatenate([obs, self._trial_phase()])
        # LAST field of the env's own observation, so the oracle wrapper's
        # chem_gt tail (appended after this) still sits at the very end and
        # every existing `context_dim`-based slice keeps working. The agent
        # excises this block before anything sees the observation.
        if self.aux_canon_target:
            obs = np.concatenate([obs, self._aux_canon_targets()])
        context = np.asarray(ts.observation[_CHEM_KEY], dtype=np.float32)
        if self.context_graph_only:
            context = context[:12]  # dims 0-11 = graph; 12-27 = frame maps
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
        if a is None:
            return "-"
        decoded = decode_action(int(a))
        if decoded.kind == "no_op":
            return "no-op"
        if decoded.kind == "cash":
            return f"stone{decoded.stone_index} -> cauldron"
        return f"stone{decoded.stone_index} -> potion{decoded.potion_index}"

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
