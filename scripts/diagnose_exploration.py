"""Is MATE's exploration degenerate, and where does the missing return go?

Two numbers the project has argued about without measuring.

**Repeat rate.** §5.2 concluded "experimenting is not scarce" from potion usage
(~86% of slots consumed). But spending a potion is not the same as learning
from it: re-applying the same perceived potion type to a stone already at that
perceived position returns information the agent already had. This counts the
fraction of potion applications whose (perceived stone coords, perceived potion
type) pair had already been tried earlier IN THE SAME EPISODE.

**Where the score goes.** §8 deferred logging the value of cashed stones. The
gap to the ceiling is either +15 stones never cashed or -1 stones cashed
anyway, and those imply different fixes. This reports the latent value of every
stone cashed, and of every stone still held when a trial resets.

The trained policy is compared against uniform-over-legal on the same
checkpoint, so "degenerate" is measured against a policy that is definitionally
not strategic.

    python scripts/diagnose_exploration.py --run logs/count/*/mate_probe_ctrl_*/
"""
import argparse, collections, glob, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from ml_collections import ConfigDict
from envs.alchemy import SymbolicAlchemyEnv, decode_action
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN
from utils.checkpointing import load_training_checkpoint

LEVELS = {
    "rotation_random_bottleneck":
        "perceptual_mapping_randomized_with_rotation_and_random_bottleneck",
}


def build(run_dir):
    ckpt = load_training_checkpoint(
        os.path.join(run_dir, "training_checkpoint.pth"), map_location="cpu"
    )
    cfg = ckpt["config"]
    ce, cr, cs = (ConfigDict(cfg[k]) for k in ("config_env", "config_rl", "config_seq"))
    env_kw = dict(
        level_name=LEVELS[ce.env_name], num_trials=ce.num_trials,
        max_steps_per_trial=ce.max_steps_per_trial,
        observe_used=ce.observe_used, add_trial_flag=ce.add_trial_flag,
        structured_potions=ce.structured_potions,
        aux_canon_target=ce.aux_canon_target,
    )
    probe_env = SymbolicAlchemyEnv(**env_kw)
    agent = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=probe_env.observation_space.shape[0],
        action_dim=probe_env.action_space.n,
        config_seq=cs, config_rl=cr, config_env=ce, freeze_critic=False,
    ).to(ptu.device)
    agent.load_state_dict(ckpt["agent_training_state"]["model"])
    agent.eval()
    print(f"loaded {run_dir}")
    return agent, ce, env_kw


def stone_state(env):
    """slot -> (perceived coord tuple, latent reward) for stones present now."""
    from dm_alchemy.types import stones_and_potions as sp

    dm = env._env
    chem = dm._chemistry
    out = {}
    for s in dm.game_state.existing_stones():
        slot = dm.game_state.get_stone_ind(stone_inst=s.idx)
        perceived = sp.unalign(
            chem.stone_map.apply_inverse(s.latent_stone()), chem.rotation
        )
        coords = tuple(np.rint(np.asarray(perceived.perceived_coords, float)).astype(int))
        out[slot] = (coords, int(np.sum(np.asarray(s.latent, float))))
    return out


def chemistry_view(env):
    """(latent stone coords by slot, latent potion (dim, dir) by slot, adjacency).

    Everything needed to say WHY a potion application did nothing. A no-change
    result has two unrelated causes and the first version of this script
    conflated them:

      already  -- the stone's coordinate on that axis already equals the
                  potion's direction, so there is nothing to flip. Nothing to
                  do with the bottleneck graph, and it happens for roughly half
                  of random (stone, potion) pairs.
      blocked  -- the flip is defined but the cube edge is closed.

    Only `blocked` is evidence about the graph.
    """
    from dm_alchemy.types import graphs, stones_and_potions as sp

    dm = env._env
    chem = dm._chemistry
    stones = {
        dm.game_state.get_stone_ind(stone_inst=s.idx):
            np.rint(np.asarray(s.latent, float)).astype(int)
        for s in dm.game_state.existing_stones()
    }
    potions = {
        dm.game_state.get_potion_ind(potion_inst=p.idx):
            (int(p.dimension), 1 if p.direction > 0 else -1)
        for p in dm.game_state.existing_potions()
    }
    adj = np.asarray(graphs.convert_graph_to_adj_mat(chem.graph))
    coord_to_idx = {
        tuple(np.rint(np.asarray(sp.index_to_coords(i), float)).astype(int)): i
        for i in range(8)
    }
    return stones, potions, adj, coord_to_idx, graphs.NO_EDGE


def classify_potion_use(stone_latent, potion, adj, coord_to_idx, no_edge):
    """-> 'already' | 'blocked' | 'effective'."""
    dim, direction = potion
    if stone_latent[dim] == direction:
        return "already"
    target = stone_latent.copy()
    target[dim] = direction
    i = coord_to_idx.get(tuple(stone_latent))
    j = coord_to_idx.get(tuple(target))
    if i is None or j is None:
        return "effective"
    return "blocked" if adj[i, j] == no_edge else "effective"


def blocked_edge_count(env):
    """Closed edges out of the cube's 12.

    Only vertex pairs at Hamming distance 1 are cube EDGES; the other 16 of the
    28 pairs are not adjacent at all and also read as NO_EDGE, which is why the
    first version of this reported 18.8 out of 12.
    """
    from dm_alchemy.types import graphs, stones_and_potions as sp
    adj = np.asarray(graphs.convert_graph_to_adj_mat(env._env._chemistry.graph))
    coords = [np.rint(np.asarray(sp.index_to_coords(i), float)).astype(int)
              for i in range(8)]
    n = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.abs(coords[i] - coords[j]).sum()) == 2:   # one axis flipped
                n += int(adj[i, j] == graphs.NO_EDGE)
    return n


def potion_types(env):
    """slot -> perceived potion type index, for potions present now."""
    dm = env._env
    chem = dm._chemistry
    return {
        dm.game_state.get_potion_ind(potion_inst=p.idx):
            chem.potion_map.apply_inverse(p.latent_potion()).index()
        for p in dm.game_state.existing_potions()
    }


def run(agent, ce, env_kw, n_episodes, random_policy, seed0):
    T = ce.num_trials * ce.max_steps_per_trial
    A = agent.action_dim
    stat = collections.Counter()
    cashed = collections.Counter()
    abandoned = collections.Counter()

    for ep in range(n_episodes):
        env = SymbolicAlchemyEnv(**env_kw)
        obs, _ = env.reset(seed=seed0 + ep)
        stat["blocked_edges_total"] += blocked_edge_count(env)
        stat["episodes"] += 1
        internal, tried = None, set()
        prev_a = torch.zeros((1, A), device=ptu.device)
        prev_r = torch.zeros((1, 1), device=ptu.device)
        prev_o = ptu.from_numpy(obs[None])
        trial = 0

        for t in range(T):
            if t // ce.max_steps_per_trial != trial:       # trial rolled over
                trial = t // ce.max_steps_per_trial
                tried.clear()

            cur = ptu.from_numpy(obs[None])
            with torch.no_grad():
                a, internal = agent.act(
                    prev_internal_state=internal, prev_action=prev_a,
                    prev_reward=prev_r, prev_obs=prev_o, obs=cur,
                    deterministic=True, initial=(t == 0), timestep=t,
                )
                if random_policy:
                    a = agent.sample_random_action(raw_obs=cur)
            idx = int(a.argmax(dim=-1).item())
            d = decode_action(idx, observe_used=ce.observe_used)

            stones, potions = stone_state(env), potion_types(env)
            if d.kind == "potion" and d.stone_index in stones and d.potion_index in potions:
                key = (stones[d.stone_index][0], potions[d.potion_index])
                stat["potion_uses"] += 1
                if key in tried:
                    stat["repeat_uses"] += 1
                tried.add(key)
                lat_stones, lat_potions, adj, c2i, no_edge = chemistry_view(env)
                if d.stone_index in lat_stones and d.potion_index in lat_potions:
                    stat[classify_potion_use(
                        lat_stones[d.stone_index], lat_potions[d.potion_index],
                        adj, c2i, no_edge,
                    )] += 1
                before = stones[d.stone_index][0]
            elif d.kind == "cash" and d.stone_index in stones:
                cashed[stones[d.stone_index][1]] += 1
                stat["cash"] += 1
                before = None
            else:
                stat["no_op_or_invalid"] += 1
                before = None

            last_step_of_trial = (t + 1) % ce.max_steps_per_trial == 0
            if last_step_of_trial:
                # BEFORE env.step refreshes the trial: whatever is still held
                # here is lost. Counting after the step would count the NEXT
                # trial's fresh stones instead (the bug in the first version).
                for _, (_, reward) in stone_state(env).items():
                    abandoned[reward] += 1

            obs, r, term, trunc, _ = env.step(idx)
            if before is not None:
                after = stone_state(env).get(d.stone_index, (before, 0))[0]
                if after == before:
                    stat["no_change"] += 1
            prev_a, prev_r, prev_o = a, ptu.from_numpy(
                np.asarray([[r]], dtype=np.float32)
            ), cur
            if term or trunc:
                break
    return stat, cashed, abandoned


def report(tag, stat, cashed, abandoned, n_episodes):
    uses = max(stat["potion_uses"], 1)
    print(f"\n--- {tag} ({n_episodes} episodes) ---")
    print(f"  potion applications / episode : {stat['potion_uses']/n_episodes:6.1f}")
    print(f"  REPEAT rate (pair already tried this trial) : "
          f"{100*stat['repeat_uses']/uses:5.1f}%")
    print(f"  no-change results (any cause)              : "
          f"{100*stat['no_change']/uses:5.1f}%")
    print(f"     ...because ALREADY at that value        : "
          f"{100*stat['already']/uses:5.1f}%   (nothing to do with the graph)")
    print(f"     ...because the EDGE IS BLOCKED          : "
          f"{100*stat['blocked']/uses:5.1f}%   <- the only graph evidence")
    print(f"  effective (the stone actually moved)       : "
          f"{100*stat['effective']/uses:5.1f}%")
    print(f"  blocked edges in the level  : "
          f"{stat['blocked_edges_total']/max(stat['episodes'],1):4.1f} / 12 per episode")
    print(f"  no-op / invalid steps / episode            : "
          f"{stat['no_op_or_invalid']/n_episodes:6.1f}")
    print(f"  stones cashed / episode                    : {stat['cash']/n_episodes:6.1f} / 30")
    tot_c = sum(cashed.values()) or 1
    print("  cashed stone values : " + "  ".join(
        f"{v:+d}:{100*cashed[v]/tot_c:4.1f}%" for v in sorted(cashed)))
    tot_a = sum(abandoned.values()) or 1
    print("  ABANDONED at trial end: " + "  ".join(
        f"{v:+d}:{100*abandoned[v]/tot_a:4.1f}%" for v in sorted(abandoned))
          + f"   ({sum(abandoned.values())/n_episodes:.1f}/episode)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--seed", type=int, default=9000)
    ap.add_argument("--device", type=int, default=3)
    args = ap.parse_args()

    ptu.set_gpu_mode(torch.cuda.is_available(), args.device)
    run_dir = sorted(glob.glob(args.run))[0].rstrip("/")
    agent, ce, env_kw = build(run_dir)
    for tag, rnd in (("TRAINED policy", False), ("uniform-over-legal", True)):
        stat, cashed, abandoned = run(
            agent, ce, env_kw, args.episodes, rnd, args.seed + (0 if rnd else 5000)
        )
        report(tag, stat, cashed, abandoned, args.episodes)
