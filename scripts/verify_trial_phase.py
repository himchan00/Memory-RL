"""Is add_trial_phase lossless, correct, and non-privileged?

It is now on for every baseline, so the three ways it could quietly be wrong
are worth checking rather than assuming:

  loss        it must APPEND two channels, not rewrite or reorder the existing
              observation. A silent shift would move every downstream slice --
              the action mask and the aux label both index by position.
  correctness the two values must actually be (steps_left_in_trial,
              trials_left), resetting on the trial boundary. An off-by-one here
              is invisible in the return and would teach the agent the wrong
              deadline.
  privilege   it must be a function of the agent's own clock alone. If it
              varied with the hidden chemistry it would be a leak, and every
              memory-model baseline would be contaminated.

Two envs are stepped in LOCKSTEP on the same seed and action sequence, one with
the flag and one without, so nothing about the observation pipeline is
re-implemented here.

    python scripts/verify_trial_phase.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from configs.envs.alchemy import LEVELS
from envs.alchemy import (
    SymbolicAlchemyEnv, valid_action_mask_from_observation,
)

PASS, FAIL = "  PASS", "  FAIL"
results = []
TRIALS, STEPS = 10, 20


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


def make(tp):
    return SymbolicAlchemyEnv(
        level_name=LEVELS["no_bottleneck"], num_trials=TRIALS,
        max_steps_per_trial=STEPS, observe_used=True, add_trial_flag=True,
        structured_potions=True, add_trial_phase=tp,
    )


def main():
    off, on = make(False), make(True)
    n_ep = 8
    width_ok = prefix_ok = value_ok = mask_ok = True
    ctx_same = True
    seen_first, seen_last = [], []

    for ep in range(n_ep):
        seed = 6100 + ep
        o0, i0 = off.reset(seed=seed)
        o1, i1 = on.reset(seed=seed)
        rng = np.random.default_rng(seed)
        ctx_same &= np.array_equal(np.asarray(i0["context"]),
                                   np.asarray(i1["context"]))
        t = 0
        while True:
            a0, a1 = np.asarray(o0, float), np.asarray(o1, float)
            width_ok &= (a1.shape[0] == a0.shape[0] + 2)
            # the original observation must survive untouched, as a PREFIX
            prefix_ok &= np.allclose(a1[:a0.shape[0]], a0)
            # and the two appended values must be exactly the schedule
            within = t % STEPS
            want = np.array([(STEPS - within) / STEPS,
                             max((TRIALS - t // STEPS) / TRIALS, 0.0)])
            value_ok &= np.allclose(a1[-2:], want, atol=1e-6)
            if within == 0:
                seen_first.append(float(a1[-2]))
            if within == STEPS - 1:
                seen_last.append(float(a1[-2]))
            # the action mask must be unaffected
            m0 = np.asarray(valid_action_mask_from_observation(
                a0.astype(np.float32), observe_used=True, add_trial_flag=True,
                structured_potions=True, add_trial_phase=False))
            m1 = np.asarray(valid_action_mask_from_observation(
                a1.astype(np.float32), observe_used=True, add_trial_flag=True,
                structured_potions=True, add_trial_phase=True))
            mask_ok &= np.array_equal(m0, m1)

            act = int(rng.choice(np.flatnonzero(m0)))
            o0, r0, d0, u0, _ = off.step(act)
            o1, r1, d1, u1, _ = on.step(act)
            if not (np.isclose(r0, r1) and d0 == d1 and u0 == u1):
                check("rewards/termination identical", False, f"step {t}")
                return
            t += 1
            if d0 or u0:
                break
    off.close(); on.close()

    check("appends exactly 2 channels", width_ok)
    check("original observation is an untouched PREFIX (no loss, no reorder)",
          prefix_ok)
    check("values equal (steps_left/20, trials_left/10) at every step", value_ok)
    check("action mask is unchanged by the flag", mask_ok)
    check("oracle context is unchanged", ctx_same)
    check("resets on every trial boundary",
          len(seen_first) == n_ep * TRIALS and all(abs(v - 1.0) < 1e-6 for v in seen_first)
          and all(abs(v - 0.05) < 1e-6 for v in seen_last),
          f"first step {set(np.round(seen_first,3))}, last step {set(np.round(seen_last,3))}")

    # --- privilege: same t must give the same value under DIFFERENT chemistry
    vals = []
    for seed in (11, 22, 33, 44):
        e = make(True)
        o, _ = e.reset(seed=seed)
        rng = np.random.default_rng(seed)
        row = []
        for _ in range(40):
            row.append(np.asarray(o, float)[-2:].copy())
            m = np.asarray(valid_action_mask_from_observation(
                np.asarray(o, np.float32), observe_used=True, add_trial_flag=True,
                structured_potions=True, add_trial_phase=True))
            o, *_ = e.step(int(rng.choice(np.flatnonzero(m))))
        e.close()
        vals.append(np.stack(row))
    same = all(np.allclose(vals[0], v) for v in vals[1:])
    check("NON-PRIVILEGED: identical across 4 different chemistries", same,
          "a function of the agent's clock only")

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
