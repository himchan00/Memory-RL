#!/usr/bin/env python
"""Measure the scripted Alchemy baselines under the standard eval protocol.

Establishes the two numbers a learned agent should be read against:
``random_stone_potion`` (the no-chemistry floor) and ``chemistry_oracle`` (the
ceiling reachable with perfect chemistry knowledge). A run's ``eval/return``
means little on its own -- Alchemy returns are large and dominated by rewards
the agent gets for free -- so this script also reports the NORMALIZED SCORE::

    normalized = (agent - random_stone_potion) / (chemistry_oracle - random_stone_potion)

0.0 = no better than random potion-shuffling, 1.0 = optimal play.

Protocol: ``--num_episodes`` episodes split evenly over ``--n_env`` worker
processes seeded ``--seed_base .. --seed_base + n_env - 1``, matching the eval
vector env in ``main.py``. Adaptation = mean(trials 7-9) - mean(trials 0-2),
the within-episode learning signal; the scripted policies score ~0 by
construction, so any positive value from an agent is memory doing work.

Requires the dm_alchemy special install (``scripts/install_dm_alchemy.sh``).

Examples::

    # Reference table (~1 min for 1024 episodes on 16 procs)
    python scripts/eval_alchemy.py --out logs/alchemy_baselines.json

    # Position a training run against it
    python scripts/eval_alchemy.py --compare oracle_markov=161.2 --compare mate=152.4
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.envs.alchemy import LEVELS  # noqa: E402
from envs.alchemy import SymbolicAlchemyEnv  # noqa: E402
from envs.alchemy_baselines import POLICIES, rollout_episode, summarize  # noqa: E402

FLOOR_POLICY = "random_stone_potion"
CEILING_POLICY = "chemistry_oracle"


def _run_shard(job):
    """One worker: build an env, roll out ``num_episodes``, return raw arrays."""
    policy_name, seed, num_episodes, env_kwargs = job
    policy_cls = POLICIES[policy_name]
    # Seed only the policies that take one; the oracle is deterministic.
    try:
        policy = policy_cls(seed=seed)
    except TypeError:
        policy = policy_cls()

    env = SymbolicAlchemyEnv(**env_kwargs)
    returns = np.empty(num_episodes)
    per_trial = np.empty((num_episodes, env.num_trials))
    for episode in range(num_episodes):
        # Only the first reset is seeded: dm_alchemy draws a fresh chemistry
        # from the env RNG on each unseeded reset, so one seed fixes the shard.
        returns[episode], per_trial[episode] = rollout_episode(
            env, policy, seed=seed if episode == 0 else None)
    return returns, per_trial


def evaluate_policy(policy_name, *, num_episodes, n_env, seed_base, env_kwargs,
                    max_workers=None):
    per_shard = _split_evenly(num_episodes, n_env)
    jobs = [
        (policy_name, seed_base + shard, count, env_kwargs)
        for shard, count in enumerate(per_shard) if count > 0
    ]
    if len(jobs) == 1:
        results = [_run_shard(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=max_workers or len(jobs)) as pool:
            results = list(pool.map(_run_shard, jobs))
    returns = np.concatenate([r for r, _ in results])
    per_trial = np.concatenate([p for _, p in results])
    return summarize(returns, per_trial, name=policy_name)


def _split_evenly(total, parts):
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _parse_compare(entries):
    """``name=return`` or ``name=return:adaptation`` -> list of dicts."""
    parsed = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"--compare expects name=return[:adaptation], got {entry!r}")
        name, value = entry.split("=", 1)
        head, _, adaptation = value.partition(":")
        parsed.append({
            "name": name,
            "return_mean": float(head),
            "adaptation_mean": float(adaptation) if adaptation else None,
        })
    return parsed


def _print_table(rows, floor, ceiling):
    span = ceiling - floor
    print(f"\n{'policy':>22}  {'return':>16}  {'adaptation':>12}  {'normalized':>10}")
    print("-" * 68)
    for row in rows:
        stderr = row.get("return_stderr")
        ret = (f"{row['return_mean']:8.2f} +/- {stderr:5.2f}" if stderr is not None
               else f"{row['return_mean']:8.2f}          ")
        adaptation = row.get("adaptation_mean")
        adapt = f"{adaptation:+8.2f}    " if adaptation is not None else " " * 12
        normalized = (row["return_mean"] - floor) / span if abs(span) > 1e-8 else float("nan")
        print(f"{row['name']:>22}  {ret}  {adapt}  {normalized:10.3f}")
    print("-" * 68)
    print(f"normalized 0.0 = {FLOOR_POLICY} ({floor:.1f}), "
          f"1.0 = {CEILING_POLICY} ({ceiling:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", default="rotation_random_bottleneck",
                        choices=sorted(LEVELS), help="alias from configs/envs/alchemy.py")
    parser.add_argument("--num_episodes", type=int, default=1024)
    parser.add_argument("--n_env", type=int, default=16,
                        help="worker processes / distinct seeds (matches main.py eval)")
    parser.add_argument("--seed_base", type=int, default=100,
                        help="worker i uses seed_base + i")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--max_steps_per_trial", type=int, default=20)
    parser.add_argument("--policies", default=",".join(POLICIES),
                        help=f"comma-separated subset of {sorted(POLICIES)}")
    parser.add_argument("--compare", action="append", default=[],
                        help="name=return[:adaptation]; positions a run in the table")
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args()

    env_kwargs = dict(
        level_name=LEVELS[args.level],
        num_trials=args.num_trials,
        max_steps_per_trial=args.max_steps_per_trial,
        observe_used=True,
        add_trial_flag=True,
    )
    policy_names = [name.strip() for name in args.policies.split(",") if name.strip()]
    unknown = set(policy_names) - set(POLICIES)
    if unknown:
        parser.error(f"unknown policies {sorted(unknown)}; choose from {sorted(POLICIES)}")

    print(f"level={args.level}  episodes={args.num_episodes}  "
          f"seeds={args.seed_base}..{args.seed_base + args.n_env - 1}")
    results = {}
    for name in policy_names:
        results[name] = evaluate_policy(
            name,
            num_episodes=args.num_episodes,
            n_env=args.n_env,
            seed_base=args.seed_base,
            env_kwargs=env_kwargs,
        )
        summary = results[name]
        print(f"  {name:>22}  {summary['return_mean']:8.2f} "
              f"+/- {summary['return_stderr']:.2f}")

    rows = list(results.values()) + _parse_compare(args.compare)
    if FLOOR_POLICY in results and CEILING_POLICY in results:
        _print_table(rows, results[FLOOR_POLICY]["return_mean"],
                     results[CEILING_POLICY]["return_mean"])
    else:
        print(f"\n(run both {FLOOR_POLICY} and {CEILING_POLICY} "
              "to get normalized scores)")
        for row in rows:
            print(f"  {row['name']:>22}  {row['return_mean']:8.2f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"level": args.level, "num_episodes": args.num_episodes,
                       "seed_base": args.seed_base, "n_env": args.n_env,
                       "results": results}, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
