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
  uninformative     no-op, cash-in, or an action on an already-used potion or
                    stone.

"Facts" counts DISTINCT potion types first revealed, which is the number the
memory actually has to hold; "map-informative" counts every transition that
carries one, redundancy included.

Two policies, because the answer depends on who is acting: a uniform random
policy wastes most of its actions on invalid slots, while random_stone_potion
(the paper's RandomActionBot, the no-chemistry floor) always picks a live
stone and a live potion.

    python scripts/count_informative.py
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.alchemy import SymbolicAlchemyEnv, decode_action
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
        env.reset(seed=int(seed0 + ep))
        dm = env._env
        pol = None
        if policy_kind == "random_stone_potion":
            pol = RandomStonePotionPolicy(seed=int(seed0 + ep))
            pol.reset()
        seen_types, n_map, n_graph, n_other, n_steps = set(), 0, 0, 0, 0
        while True:
            before_s, before_p = state(dm)
            act = pol.act(dm) if pol is not None else int(rng.integers(n_act))
            d = decode_action(act)
            _, _, term, trunc, _ = env.step(act)
            after_s, _ = state(dm)
            n_steps += 1
            if d.kind == "potion" and d.stone_index in before_s \
                    and d.potion_index in before_p:
                moved = not np.allclose(
                    after_s.get(d.stone_index, before_s[d.stone_index]),
                    before_s[d.stone_index])
                if moved:
                    n_map += 1
                    seen_types.add(before_p[d.potion_index])
                else:
                    n_graph += 1
            else:
                n_other += 1
            if term or trunc:
                break
        tot["steps"] += n_steps; tot["map"] += n_map
        tot["graph"] += n_graph; tot["other"] += n_other
        facts_per_ep.append(len(seen_types)); mapinfo_per_ep.append(n_map)
    env.close()
    n = episodes
    return dict(
        steps=tot["steps"]/n, map=tot["map"]/n, graph=tot["graph"]/n,
        other=tot["other"]/n,
        facts=float(np.mean(facts_per_ep)), facts_sd=float(np.std(facts_per_ep)),
        mapinfo_sd=float(np.std(mapinfo_per_ep)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000)
    a = ap.parse_args()
    print(f"{a.episodes} episodes per row, 10 trials x 20 steps = 200 transitions\n")
    print(f"{'policy':22s} {'map-inf':>9s} {'graph-inf':>10s} {'uninf':>8s} "
          f"{'distinct facts':>15s}")
    print("-" * 70)
    for pk in ("uniform_random", "random_stone_potion"):
        r = run("perceptual_mapping_randomized", pk, a.episodes, a.seed)
        print(f"{pk:22s} {r['map']:9.1f} {r['graph']:10.1f} {r['other']:8.1f} "
              f"{r['facts']:9.1f} +- {r['facts_sd']:.1f}")
    print("\n  map-inf   potion applied, stone's latent coords moved -> pins one "
          "potion type's axis+direction")
    print("  graph-inf potion applied, nothing moved -> the edge is blocked "
          "(constrains the graph, not the map)")
    print("  facts     DISTINCT potion types ever revealed; 12 types exist "
          "(6 axes x 2 directions)")


if __name__ == "__main__":
    main()
