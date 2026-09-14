"""Can a plain MLP imitate the chemistry planner from the oracle's own input?

The RL oracle (markov + is_oracle) sees exactly ``concat(symbolic_obs, chem_gt)``
-- 68 dims -- and stalls far below the scripted planner. Two things could explain
that: the input is hard to DECODE into an action (representation / capacity), or
it is decodable and the RL objective simply fails to find the policy
(optimization). This script settles it by removing RL entirely: it collects
(obs, planner action) pairs and fits them with supervised cross-entropy.

    high accuracy + high BC return  -> representation is fine, RL is the culprit
    small net fails, large succeeds -> decoder capacity
    both fail                       -> the 68-dim input really is hard to decode

Train/test are split BY EPISODE: every step of an episode shares one chemistry,
so a per-step split would leak the answer across the boundary.

    python scripts/bc_diagnostic.py --train_episodes 2000 --eval_episodes 256
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
NUM_TRIALS = 10
STEPS_PER_TRIAL = 20
NO_OP_ACTION = 0


def make_env():
    from envs.alchemy import SymbolicAlchemyEnv

    return SymbolicAlchemyEnv(
        LEVEL,
        num_trials=NUM_TRIALS,
        max_steps_per_trial=STEPS_PER_TRIAL,
        observe_used=True,
        add_trial_flag=True,
    )


def oracle_obs(obs, info):
    """Exactly what ``oracleWrapper`` hands the agent: obs with chem_gt appended."""
    return np.concatenate([obs, info["context"]], axis=-1).astype(np.float32)


def latent_obs(env, obs, info):
    """The same 68-dim vector, but with the perceptual frame already undone.

    Only two things change: each stone's coordinate triple becomes its LATENT
    coordinates, and each potion's type value becomes its LATENT (axis,
    direction) type. Rewards, used-flags, absent-slot sentinels, the trial flag
    and the appended chem_gt are byte-identical to ``oracle_obs``. So the single
    variable between the two conditions is the frame -- this is exactly what a
    ``canonicalize_oracle`` preprocessing step would hand the network, and the
    chem_gt tail is kept (now redundant) so dimensionality matches too.
    """
    dm = env._env
    out = np.array(obs, dtype=np.float32, copy=True)
    for stone in dm.game_state.existing_stones():
        slot = dm.game_state.get_stone_ind(stone_inst=stone.idx)
        out[5 * slot:5 * slot + 3] = np.asarray(stone.latent, dtype=np.float32)
    for potion in dm.game_state.existing_potions():
        slot = dm.game_state.get_potion_ind(potion_inst=potion.idx)
        latent_type = int(potion.dimension) * 2 + (1 if potion.direction > 0 else 0)
        # Same scalar encoding the env uses for the perceived type: index/3 - 1.
        out[15 + 2 * slot] = latent_type / 3.0 - 1.0
    return np.concatenate([out, info["context"]], axis=-1).astype(np.float32)


# --------------------------------------------------------------- data collection
def collect_worker(job):
    """Roll the planner out for ``n`` episodes; return (obs, actions, returns)."""
    seed, n = job
    from envs.alchemy_baselines import ChemistryOraclePolicy

    env = make_env()
    policy = ChemistryOraclePolicy()
    obs_buf, lat_buf, act_buf, ret_buf = [], [], [], []

    for episode in range(n):
        policy.reset()
        # dm_alchemy draws a fresh chemistry from the env RNG on every unseeded
        # reset, so seeding only the first episode still gives n distinct ones.
        o, info = env.reset(seed=seed if episode == 0 else None)
        total = 0.0
        for _ in range(env.max_episode_steps):
            # Both frames of the SAME state, so the two conditions are paired.
            obs_buf.append(oracle_obs(o, info))
            lat_buf.append(latent_obs(env, o, info))
            action = policy.act(env)
            act_buf.append(action)
            o, reward, _, truncated, info = env.step(action)
            total += reward
            if truncated:
                break
        ret_buf.append(total)

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(lat_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.int64),
        np.asarray(ret_buf, dtype=np.float64),
    )


def collect(num_episodes, num_workers, base_seed):
    per = [num_episodes // num_workers] * num_workers
    for i in range(num_episodes % num_workers):
        per[i] += 1
    jobs = [(base_seed + i, n) for i, n in enumerate(per) if n > 0]

    start = time.time()
    if len(jobs) == 1:
        parts = [collect_worker(jobs[0])]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            parts = pool.map(collect_worker, jobs)
    elapsed = time.time() - start

    obs = np.concatenate([p[0] for p in parts])
    lat = np.concatenate([p[1] for p in parts])
    act = np.concatenate([p[2] for p in parts])
    rets = np.concatenate([p[3] for p in parts])
    # Episode id per step, so the split can be made along episode boundaries.
    ep_ids, offset = [], 0
    for p in parts:
        n_ep = len(p[3])
        ep_ids.append(offset + np.arange(len(p[2])) // (NUM_TRIALS * STEPS_PER_TRIAL))
        offset += n_ep
    ep_ids = np.concatenate(ep_ids)
    print(f"  collected {len(rets)} episodes / {len(act)} steps in {elapsed:.0f}s "
          f"(planner return {rets.mean():.1f})")
    return obs, lat, act, ep_ids, rets


# ---------------------------------------------------------------------- models
def build_mlp(in_dim, hidden, out_dim):
    import torch.nn as nn

    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def train_bc(model, x_tr, y_tr, x_te, y_te, epochs, batch_size, lr, tag):
    import torch
    import torch.nn.functional as F

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(x_tr)
    g = torch.Generator().manual_seed(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        loss_sum = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
        sched.step()
        if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            acc = evaluate_accuracy(model, x_te, y_te)
            print(f"    [{tag}] epoch {epoch:3d}  train loss {loss_sum / n:.4f}  "
                  f"test acc {acc['all']:.3f}  (non-noop {acc['action']:.3f})")
    return model


def evaluate_accuracy(model, x, y):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), 8192):
            preds.append(model(x[i:i + 8192]).argmax(-1))
    pred = torch.cat(preds)
    correct = (pred == y).float()
    real = y != NO_OP_ACTION
    return {
        "all": correct.mean().item(),
        "action": correct[real].mean().item() if real.any() else float("nan"),
        "noop_frac": (~real).float().mean().item(),
    }


# ------------------------------------------------------------------ env rollout
def rollout_bc(model, num_episodes, seed, context_dim, frame="perceived"):
    """Greedy rollout of the BC net, with the same invalid-action masking as DQN.

    The mask is always derived from the PERCEIVED observation: it reads only the
    used-flags, which the latent rewrite leaves untouched, but deriving it from
    the real observation keeps the two conditions on identical footing.
    """
    import torch
    from envs.alchemy import valid_action_mask_from_observation

    env = make_env()
    returns = np.empty(num_episodes)
    per_trial = np.zeros((num_episodes, NUM_TRIALS))
    model.eval()

    for episode in range(num_episodes):
        o, info = env.reset(seed=seed if episode == 0 else None)
        for step in range(env.max_episode_steps):
            perceived = torch.from_numpy(oracle_obs(o, info)).unsqueeze(0)
            net_in = (perceived if frame == "perceived"
                      else torch.from_numpy(latent_obs(env, o, info)).unsqueeze(0))
            with torch.no_grad():
                mask = valid_action_mask_from_observation(
                    perceived,
                    observe_used=True,
                    add_trial_flag=True,
                    context_dim=context_dim,
                )
                logits = model(net_in).masked_fill(~mask, -torch.inf)
                action = int(logits.argmax(-1).item())
            o, reward, _, truncated, info = env.step(action)
            per_trial[episode, min(step // STEPS_PER_TRIAL, NUM_TRIALS - 1)] += reward
            if truncated:
                break
        returns[episode] = per_trial[episode].sum()
    return returns, per_trial


# ------------------------------------------------------------------------ main
FLOOR, CEILING = 145.2, 287.1


def normalized(x):
    return (x - FLOOR) / (CEILING - FLOOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_episodes", type=int, default=2000)
    ap.add_argument("--test_episodes", type=int, default=200)
    ap.add_argument("--eval_episodes", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.seed)

    print("=" * 78)
    print("BC DIAGNOSTIC -- can supervised learning decode the oracle's input?")
    print("=" * 78)
    print(f"\n[1] collecting planner demonstrations "
          f"({args.train_episodes} train + {args.test_episodes} test episodes)")
    obs, lat, act, ep_ids, rets = collect(
        args.train_episodes + args.test_episodes, args.workers, args.seed)

    n_test_ep = args.test_episodes
    max_ep = ep_ids.max() + 1
    is_test = ep_ids >= (max_ep - n_test_ep)
    y_tr = torch.from_numpy(act[~is_test])
    y_te = torch.from_numpy(act[is_test])
    in_dim, out_dim = obs.shape[1], 40
    context_dim = 28
    print(f"  train {int((~is_test).sum())} steps / test {int(is_test.sum())} steps  "
          f"(in_dim {in_dim}, {out_dim} actions, "
          f"no-op fraction {(act == NO_OP_ACTION).mean():.2f})")
    frac_differ = float((obs != lat).any(axis=1).mean())
    print(f"  latent frame differs from perceived on {frac_differ:.1%} of steps")

    configs = [
        # Capacity-matched to the RL oracle's decoder: conditioner (128 wide,
        # 1 layer) feeding a (256, 256) critic MLP.
        ("small", (128, 256, 256)),
        ("large", (1024, 1024, 1024, 1024)),
    ]
    # Paired A/B on the SAME episodes: only the frame of the input differs.
    frames = [("perceived", obs), ("latent", lat)]

    results = {}
    for frame, data in frames:
        x_tr = torch.from_numpy(data[~is_test])
        x_te = torch.from_numpy(data[is_test])
        for tag, hidden in configs:
            key = f"{frame}/{tag}"
            model = build_mlp(in_dim, hidden, out_dim)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n[2] training '{key}' MLP {in_dim}->{hidden}->{out_dim} "
                  f"({n_params/1e3:.0f}k params)")
            t0 = time.time()
            train_bc(model, x_tr, y_tr, x_te, y_te,
                     args.epochs, args.batch_size, args.lr, key)
            acc = evaluate_accuracy(model, x_te, y_te)
            print(f"  trained in {time.time() - t0:.0f}s")

            print(f"\n[3] rolling '{key}' out in the env "
                  f"({args.eval_episodes} episodes, invalid-action masking on)")
            t0 = time.time()
            bc_ret, bc_trial = rollout_bc(
                model, args.eval_episodes, args.seed + 777, context_dim, frame=frame)
            early, late = bc_trial[:, :3].mean(), bc_trial[:, 7:].mean()
            print(f"  return {bc_ret.mean():.1f} "
                  f"+- {bc_ret.std()/np.sqrt(len(bc_ret)):.1f}"
                  f"   normalized {normalized(bc_ret.mean()):.3f}"
                  f"   adaptation {late - early:+.2f}   ({time.time() - t0:.0f}s)")
            results[key] = dict(acc=acc, ret=bc_ret.mean(),
                                adapt=late - early, params=n_params)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'frame/model':<20}{'params':>10}{'test acc':>10}{'non-noop':>10}"
          f"{'BC return':>12}{'normalized':>12}")
    for key, r in results.items():
        print(f"{key:<20}{r['params']/1e3:>9.0f}k{r['acc']['all']:>10.3f}"
              f"{r['acc']['action']:>10.3f}{r['ret']:>12.1f}"
              f"{normalized(r['ret']):>12.3f}")
    print(f"{'-'*74}")
    print(f"{'planner':<20}{'':>10}{1.0:>10.3f}{1.0:>10.3f}"
          f"{rets.mean():>12.1f}{normalized(rets.mean()):>12.3f}")
    print(f"{'floor':<20}{'':>10}{'':>10}{'':>10}{FLOOR:>12.1f}{0.0:>12.3f}")

    print("\nreading (the perceived-vs-latent contrast is the point):")
    print("  latent >> perceived -> the PERCEPTUAL FRAME is the bottleneck;")
    print("     canonicalizing the oracle's input is the fix, and it is cheap.")
    print("  latent ~= perceived, both far short -> the frame is NOT the issue;")
    print("     imitating the planner needs SEARCH, which no feedforward decoder")
    print("     does. Representation work will not close the gap.")


if __name__ == "__main__":
    main()
