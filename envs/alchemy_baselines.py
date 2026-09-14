"""Scripted baselines for Symbolic Alchemy, in the repo's integer action space.

Three reference policies, all emitting the same ``int`` actions the agent uses
(see :mod:`envs.alchemy`), so their returns are directly comparable to a
training run's ``eval/return``:

===================  ==========================================================
``UniformRandom``    uniform over the 40 raw actions; most are no-ops
``RandomStonePotion``the paper's ``RandomActionBot`` — random (stone, potion)
                     pairs, cash when a stone tops out or potions run out
``ChemistryOracle``  exact planner over the ground-truth chemistry, in LATENT
                     space; reproduces the paper's unreleased ``search_oracle``
===================  ==========================================================

Reference returns on ``rotation_random_bottleneck`` (10 trials x 20 steps),
under the standard eval protocol -- 1024 episodes, seeds 100..115, i.e.
``python scripts/eval_alchemy.py`` (paper values in parentheses):

    UniformRandom        17.05 +/- 0.55 (stderr)
    RandomStonePotion   145.20 +/- 1.52            (145.7)
    ChemistryOracle     287.08 +/- 1.55            (288.5)

All three score ~0 adaptation, as they must: none of them learn within an
episode. Any positive adaptation from a trained agent is memory doing work.

``ChemistryOracle`` is the ceiling a learned oracle-Markov agent should be
measured against. dm_alchemy's own ``IdealObserverBot`` is NOT a usable
substitute: at 12 potions x 3 stones its belief-state search does not finish
within 15 minutes per episode, and the recorded traces it ships were generated
on the non-rotated level only.

Why the planner lives in latent space: ``chem_gt`` and ``symbolic_obs`` are
expressed in different frames, so a perceptual-space planner would first have to
invert the rotation / dim-permutation / sign-flips. The env exposes the latent
state internally, so the planner reads it directly and searches there.

Requires the dm_alchemy special install (``scripts/install_dm_alchemy.sh``).
"""
from __future__ import annotations

import numpy as np

from envs.alchemy import NO_OP_ACTION, SymbolicAlchemyEnv

_NUM_POTION_TYPES = 6  # 3 axes x 2 directions


def unwrap_dm_env(env):
    """Accepts a :class:`SymbolicAlchemyEnv` (or a raw dm_alchemy env)."""
    return env._env if isinstance(env, SymbolicAlchemyEnv) else env


def _latent_index(coords) -> int:
    from dm_alchemy.types import stones_and_potions as sp

    return int(sp.LatentStone(np.asarray(coords)).index())


def _potion_type_index(potion) -> int:
    """(axis, direction) -> 0..5, matching ``_apply``'s decoding."""
    return int(potion.dimension) * 2 + (1 if potion.direction > 0 else 0)


class AlchemyPolicy:
    """Emits an integer action for the env's current state.

    ``reset()`` is called once per episode; ``act(env)`` once per step.
    """

    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, env) -> int:
        raise NotImplementedError


class UniformRandomPolicy(AlchemyPolicy):
    """Uniform over the raw action space (no validity masking)."""

    name = "uniform_random"

    def __init__(self, num_actions: int = 40, seed: int | None = None):
        self.num_actions = int(num_actions)
        self._rng = np.random.RandomState(seed)

    def act(self, env) -> int:
        return int(self._rng.randint(self.num_actions))


class RandomStonePotionPolicy(AlchemyPolicy):
    """The paper's ``RandomActionBot``, re-expressed in slot-based int actions.

    Applies a uniformly random potion to a uniformly random stone whose latent
    reward is below ``threshold_for_leaving``; once no such stone remains (or
    the potions run out) it cashes the positive stones one at a time, then
    no-ops out the trial. dm_alchemy's version ends the trial with a single
    macro action, which our fixed-step horizon does not have; the remaining
    steps are worthless either way, so returns match.

    Uses no chemistry knowledge beyond the per-stone reward that
    ``symbolic_obs`` already leaks, so this is a genuine no-chemistry baseline.
    """

    name = "random_stone_potion"

    def __init__(self, seed: int | None = None, threshold_for_leaving: int = 2):
        self._rng = np.random.RandomState(seed)
        self.threshold_for_leaving = int(threshold_for_leaving)

    def act(self, env) -> int:
        from dm_alchemy import symbolic_alchemy
        from dm_alchemy.types import graphs

        env = unwrap_dm_env(env)
        reward_weights = env._reward_weights
        stones = env.game_state.existing_stones()
        potions = env.game_state.existing_potions()

        improvable = [
            s for s in stones
            if reward_weights(s.latent) < self.threshold_for_leaving
        ]
        if improvable and potions:
            stone = improvable[self._rng.randint(len(improvable))]
            potion = potions[self._rng.randint(len(potions))]
            action = symbolic_alchemy.type_utils.SlotBasedAction(
                stone_ind=env.game_state.get_stone_ind(
                    stone=graphs.Node(-1, stone.latent)),
                potion_ind=env.game_state.get_potion_ind(potion=potion),
            )
            return int(symbolic_alchemy.slot_based_action_to_int(
                action, end_trial_action=False))
        return _cash_best_positive_stone(env)


class ChemistryOraclePolicy(AlchemyPolicy):
    """Exact planner over the ground-truth chemistry (latent-space DFS).

    Reproduces the paper's ``search_oracle`` (verified action-for-action over
    200 episodes). State is (sorted stone multiset, potion-type multiset); the
    value of a state is the best total reward reachable from it, memoized per
    chemistry. Slot identity is irrelevant -- only the multisets matter -- which
    is what keeps the search tractable (~130-190 ms per 200-step episode).
    """

    name = "chemistry_oracle"

    def __init__(self):
        self._adj = None

    def reset(self) -> None:
        self._adj = None

    def _reset_chemistry(self, env) -> None:
        from dm_alchemy.types import graphs, stones_and_potions as sp

        self._adj = np.asarray(
            graphs.convert_graph_to_adj_mat(env._chemistry.graph))
        self._reward_weights = env._reward_weights
        self._node_reward = np.array(
            [self._reward_weights(s.latent_coords)
             for s in sp.possible_latent_stones()],
            dtype=float,
        )
        self._cache = {}

    def _apply(self, node: int, axis: int, direction: int):
        """Latent transition; None if the potion is a no-op or edge is blocked."""
        from dm_alchemy.types import graphs, stones_and_potions as sp

        coords = np.array(sp.index_to_coords(node))
        if coords[axis] == direction:
            return None  # stone is already on that face
        coords[axis] = direction
        nxt = _latent_index(coords)
        return nxt if self._adj[node, nxt] != graphs.NO_EDGE else None

    def _value(self, stones: tuple, counts: tuple):
        """(best total reward, first move) for this stone/potion multiset."""
        key = (stones, counts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Baseline: cash every stone that is worth cashing, use no more potions.
        best = (sum(max(0.0, self._node_reward[n]) for n in stones), None)
        for potion_type in range(_NUM_POTION_TYPES):
            if counts[potion_type] == 0:
                continue
            axis, direction = potion_type // 2, (1 if potion_type % 2 else -1)
            remaining = list(counts)
            remaining[potion_type] -= 1
            remaining = tuple(remaining)
            for slot, node in enumerate(stones):
                nxt = self._apply(node, axis, direction)
                if nxt is None:
                    continue
                after = list(stones)
                after[slot] = nxt
                value, _ = self._value(tuple(sorted(after)), remaining)
                if value > best[0]:
                    best = (value, (slot, potion_type))
        self._cache[key] = best
        return best

    def act(self, env) -> int:
        from dm_alchemy import symbolic_alchemy
        from dm_alchemy.types import graphs

        env = unwrap_dm_env(env)
        if self._adj is None or env.is_new_trial():
            self._reset_chemistry(env)

        stones = env.game_state.existing_stones()
        if not stones:
            return NO_OP_ACTION
        potions = env.game_state.existing_potions()

        nodes = [_latent_index(s.latent) for s in stones]
        counts = [0] * _NUM_POTION_TYPES
        by_type: dict[int, list] = {}
        for potion in potions:
            potion_type = _potion_type_index(potion)
            counts[potion_type] += 1
            by_type.setdefault(potion_type, []).append(potion)

        # _value works on the SORTED multiset, so map its slot back to a real one.
        sorted_order = np.argsort(nodes, kind="stable")
        _, move = self._value(tuple(sorted(nodes)), tuple(counts))
        if move is None:
            return _cash_best_positive_stone(env)

        sorted_slot, potion_type = move
        stone = stones[int(sorted_order[sorted_slot])]
        potion = by_type[potion_type][0]  # potions of a type are interchangeable
        action = symbolic_alchemy.type_utils.SlotBasedAction(
            stone_ind=env.game_state.get_stone_ind(
                stone=graphs.Node(-1, stone.latent)),
            potion_ind=env.game_state.get_potion_ind(potion=potion),
        )
        return int(symbolic_alchemy.slot_based_action_to_int(
            action, end_trial_action=False))


def _cash_best_positive_stone(env) -> int:
    """Cash any stone worth positive reward, else no-op."""
    from dm_alchemy import symbolic_alchemy
    from dm_alchemy.types import graphs

    reward_weights = env._reward_weights
    for stone in env.game_state.existing_stones():
        if reward_weights(stone.latent) > 0:
            action = symbolic_alchemy.type_utils.SlotBasedAction(
                stone_ind=env.game_state.get_stone_ind(
                    stone=graphs.Node(-1, stone.latent)),
                cauldron=True,
            )
            return int(symbolic_alchemy.slot_based_action_to_int(
                action, end_trial_action=False))
    return NO_OP_ACTION


POLICIES = {
    UniformRandomPolicy.name: UniformRandomPolicy,
    RandomStonePotionPolicy.name: RandomStonePotionPolicy,
    ChemistryOraclePolicy.name: ChemistryOraclePolicy,
}


def rollout_episode(env: SymbolicAlchemyEnv, policy: AlchemyPolicy, seed=None):
    """One episode through the gym env; returns total and per-trial returns."""
    policy.reset()
    env.reset(seed=seed)
    per_trial = np.zeros(env.num_trials, dtype=np.float64)
    for step in range(env.max_episode_steps):
        _, reward, _, truncated, _ = env.step(policy.act(env))
        per_trial[min(step // env.max_steps_per_trial, env.num_trials - 1)] += reward
        if truncated:
            break
    return float(per_trial.sum()), per_trial


def evaluate(
    policy: AlchemyPolicy,
    env: SymbolicAlchemyEnv,
    num_episodes: int,
    seed: int | None = None,
):
    """Rolls ``policy`` out for ``num_episodes`` and summarizes.

    Only the FIRST episode is seeded: dm_alchemy draws a new chemistry from the
    env's RNG on every unseeded reset, so one seed fixes the whole sequence.
    Reseeding each episode would rebuild the env and replay one chemistry.
    """
    returns = np.empty(num_episodes, dtype=np.float64)
    per_trial = np.empty((num_episodes, env.num_trials), dtype=np.float64)
    for episode in range(num_episodes):
        returns[episode], per_trial[episode] = rollout_episode(
            env, policy, seed=seed if episode == 0 else None)
    return summarize(returns, per_trial, name=policy.name)


def summarize(returns, per_trial, name="policy", early=(0, 3), late=(7, 10)):
    """Return / adaptation stats. Adaptation = mean(late trials) - mean(early)."""
    returns = np.asarray(returns, dtype=np.float64)
    per_trial = np.asarray(per_trial, dtype=np.float64)
    n = len(returns)
    early_return = per_trial[:, early[0]:early[1]].mean(axis=1)
    late_return = per_trial[:, late[0]:late[1]].mean(axis=1)
    adaptation = late_return - early_return
    return {
        "name": name,
        "num_episodes": n,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_stderr": float(returns.std() / max(np.sqrt(n), 1e-8)),
        "trial_return_mean": per_trial.mean(axis=0).tolist(),
        "early_return_mean": float(early_return.mean()),
        "late_return_mean": float(late_return.mean()),
        "adaptation_mean": float(adaptation.mean()),
        "adaptation_stderr": float(adaptation.std() / max(np.sqrt(n), 1e-8)),
    }
