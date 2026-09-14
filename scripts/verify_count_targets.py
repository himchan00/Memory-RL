"""Check the "Predict: Features" count targets against the env's own state.

``count_targets_from_observation`` decodes the stone/potion slot blocks out of
the observation vector. This recomputes the same two count vectors the other
way round -- from ``game_state`` through the chemistry's perceptual maps, the
path ``scripts/trace_alchemy.py`` uses -- so agreement is a real check and not
a restatement of the decode.

Covers both potion layouts, since ``structured_potions`` changes the potion
block from one ordinal scalar to axis one-hot(3) + direction(1):

    python scripts/verify_count_targets.py
"""
import os, sys, numpy as np, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from envs.alchemy import (
    AUX_COUNT_POTION_CATEGORIES,
    AUX_COUNT_STONE_CATEGORIES,
    SymbolicAlchemyEnv,
    count_targets_from_observation,
    present_flags_from_observation,
)

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
EPISODES = 6
STEPS = 200


def reference_counts(env):
    """(stone_counts, potion_counts) from game_state, not from the obs vector."""
    from dm_alchemy.types import stones_and_potions as sp

    dm = env._env
    chem = dm._chemistry
    stone = np.zeros(AUX_COUNT_STONE_CATEGORIES, dtype=np.int64)
    potion = np.zeros(AUX_COUNT_POTION_CATEGORIES, dtype=np.int64)

    for s in dm.game_state.existing_stones():
        perceived = sp.unalign(
            chem.stone_map.apply_inverse(s.latent_stone()), chem.rotation
        )
        coords = np.asarray(perceived.perceived_coords, float)
        digits = np.rint(coords).astype(np.int64) + 1
        stone[digits[0] * 9 + digits[1] * 3 + digits[2]] += 1

    for p in dm.game_state.existing_potions():
        potion[chem.potion_map.apply_inverse(p.latent_potion()).index()] += 1

    return stone, potion


def run(structured_potions):
    env = SymbolicAlchemyEnv(
        level_name=LEVEL,
        num_trials=10,
        max_steps_per_trial=20,
        observe_used=True,
        add_trial_flag=True,
        structured_potions=structured_potions,
    )
    split = dict(
        observe_used=True,
        add_trial_flag=True,
        context_dim=0,
        structured_potions=structured_potions,
    )
    n_steps = stone_bad = potion_bad = present_bad = 0
    stone_seen = np.zeros(AUX_COUNT_STONE_CATEGORIES, dtype=np.int64)

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=2000 + ep)
        for _ in range(STEPS):
            t = torch.as_tensor(obs, dtype=torch.float32)[None]
            got_stone, got_potion = count_targets_from_observation(t, **split)
            got_stone = got_stone[0].numpy().astype(np.int64)
            got_potion = got_potion[0].numpy().astype(np.int64)
            want_stone, want_potion = reference_counts(env)

            stone_bad += int(not np.array_equal(got_stone, want_stone))
            potion_bad += int(not np.array_equal(got_potion, want_potion))
            stone_seen += got_stone

            # The totals must equal the occupancy the ACTION MASK reads, or the
            # aux loss and the mask disagree about which slots exist.
            sp_flag, pp_flag = present_flags_from_observation(t, **split)
            present_bad += int(
                got_stone.sum() != int(sp_flag.sum())
                or got_potion.sum() != int(pp_flag.sum())
            )

            n_steps += 1
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break

    tag = "structured" if structured_potions else "ordinal   "
    print(f"[{tag}] steps compared        : {n_steps}")
    print(f"[{tag}] stone count mismatches: {stone_bad}   <- must be 0")
    print(f"[{tag}] potion count mismatches: {potion_bad}  <- must be 0")
    print(f"[{tag}] present-flag mismatches: {present_bad}  <- must be 0")
    print(f"[{tag}] stone categories used  : {int((stone_seen > 0).sum())}/27")
    return stone_bad + potion_bad + present_bad


if __name__ == "__main__":
    bad = run(False) + run(True)
    print("\nOK" if bad == 0 else f"\nFAILED ({bad} mismatches)")
    sys.exit(1 if bad else 0)
