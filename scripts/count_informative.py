"""How many of Alchemy's 200 transitions actually carry the frame map?

The claim "about 12 of 200" was an assumption, so this measures it. The frame
map is what turns a PERCEIVED potion into a LATENT axis and direction, and the
only way to learn one potion's entry is to watch that potion act on a stone and
see where the stone went. So a transition is counted as:

  map-informative   a potion was applied and the stone's LATENT coords moved:
                    it pins that potion type's axis and direction.
  graph-informative a potion was applied and nothing moved: that says the edge
                    is blocked, which constrains the graph but NOT the map. It
                    is real information, which is why the gate is not keyed on
                    ||delta_obs|| -- such a gate would throw these away.
  uninformative     split three ways, because the ACTION MASK removes one of
                    them entirely in training:
                      invalid  a used stone or potion -- impossible under
                               mask_alchemy_invalid_actions=True
                      cash-in  stone to the cauldron: pays out, reveals nothing
                               new about the perceived->latent potion map
                      no-op    the mask leaves NO_OP legal (mask_no_op defaults
                               to False), and once a trial's stones are all
                               cashed there is genuinely nothing else to do

"Facts" counts DISTINCT potion types first revealed, which is the number the
memory actually has to hold; "map-informative" counts every transition that
carries one, redundancy included.

Three policies, because the answer depends entirely on who is acting.
masked_random is the one that matches training: uniform over LEGAL actions, the
same draw the agent's warm-up and epsilon-greedy use under
mask_alchemy_invalid_actions=True. Reading the unmasked rows as if they were
the training setting overstates how much of the episode is wasted.

    python scripts/count_informative.py
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.alchemy import (
    SymbolicAlchemyEnv, decode_action, valid_action_mask_from_observation,
)
from envs.alchemy_baselines import RandomStonePotionPolicy


def state(dm):
    """SLOT -> latent coords, and the live potions' latent (axis, direction).

    Keyed by slot, not instance id, because that is what the action indexes.
    """
    stones = {
        dm.game_state.get_stone_ind(stone_inst=s.idx): np.asarray(s.latent, float)
        for s in dm.game_state.existing_stones()
    }
    potions = {
        dm.game_state.get_potion_ind(potion_inst=p.idx): (p.dimension, p.direction)
        for p in dm.game_state.existing_potions()
    }
    return stones, potions


def run(level, policy_kind, episodes, seed0):
    env = SymbolicAlchemyEnv(
        level_name=level, num_trials=10, max_steps_per_trial=20,
        observe_used=True, add_trial_flag=True, structured_potions=True,
    )
    n_act = env.action_space.n
    tot = Counter()
    facts_per_ep, mapinfo_per_ep = [], []
    for ep in range(episodes):
        rng = np.random.default_rng(seed0 + ep)
        cur_obs, _ = env.reset(seed=int(seed0 + ep))
        dm = env._env
        pol = None
        if policy_kind == "random_stone_potion":
            pol = RandomStonePotionPolicy(seed=int(seed0 + ep))
            pol.reset()
        seen_types = set()
        n_map = n_graph = n_cash = n_noop = n_invalid = n_steps = 0
        obs = env.observation()["symbolic_obs"] if hasattr(env, "observation") else None
        obs = None
        while True:
            before_s, before_p = state(dm)
            if policy_kind == "masked_random":
                mask = np.asarray(valid_action_mask_from_observation(
                    np.asarray(cur_obs, dtype=np.float32),
                    observe_used=True, add_trial_flag=True,
                    structured_potions=True))
                legal = np.flatnonzero(mask)
                act = int(rng.choice(legal))
            elif pol is not None:
                act = pol.act(dm)
            else:
                act = int(rng.integers(n_act))
            d = decode_action(act)
            cur_obs, _, term, trunc, _ = env.step(act)
            after_s, _ = state(dm)
            n_steps += 1
            if d.kind == "no_op":
                n_noop += 1
            elif d.kind == "cash":
                n_cash += 1
            elif d.stone_index in before_s and d.potion_index in before_p:
                moved = not np.allclose(
                    after_s.get(d.stone_index, before_s[d.stone_index]),
                    before_s[d.stone_index])
                if moved:
                    n_map += 1
                    seen_types.add(before_p[d.potion_index])
                else:
                    n_graph += 1
            else:
                n_invalid += 1
            if term or trunc:
                break
        tot["steps"] += n_steps; tot["map"] += n_map; tot["graph"] += n_graph
        tot["cash"] += n_cash; tot["noop"] += n_noop; tot["invalid"] += n_invalid
        facts_per_ep.append(len(seen_types)); mapinfo_per_ep.append(n_map)
    env.close()
    n = episodes
    return dict(
        steps=tot["steps"]/n, map=tot["map"]/n, graph=tot["graph"]/n,
        cash=tot["cash"]/n, noop=tot["noop"]/n, invalid=tot["invalid"]/n,
        facts=float(np.mean(facts_per_ep)), facts_sd=float(np.std(facts_per_ep)),
        mapinfo_sd=float(np.std(mapinfo_per_ep)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    print(f"{a.episodes} episodes per row, 10 trials x 20 steps = 200 transitions\n")
    print(f"{'policy':22s} {'map':>6s} {'graph':>6s} {'cash':>6s} {'no-op':>6s} "
          f"{'invalid':>8s} {'facts':>7s}")
    print("-" * 70)
    for pk in ("masked_random", "random_stone_potion", "uniform_random"):
        r = run("perceptual_mapping_randomized", pk, a.episodes, a.seed)
        tag = pk + (" *" if pk == "masked_random" else "")
        print(f"{tag:22s} {r['map']:6.1f} {r['graph']:6.1f} {r['cash']:6.1f} "
              f"{r['noop']:6.1f} {r['invalid']:8.1f} {r['facts']:7.1f}")
    print("\n  * masked_random is the training setting: uniform over LEGAL "
          "actions, as under\n    mask_alchemy_invalid_actions=True. The mask "
          "removes the `invalid` column; NO_OP and\n    cash-in stay legal "
          "(mask_no_op defaults to False).")
    print("\n  map-inf   potion applied, stone's latent coords moved -> pins one "
          "potion type's axis+direction")
    print("  graph-inf potion applied, nothing moved -> the edge is blocked "
          "(constrains the graph, not the map)")
    print("  facts     DISTINCT potion types ever revealed; 12 types exist "
          "(6 axes x 2 directions)")


if __name__ == "__main__":
    main()
