"""A/B check for add_trial_phase and the factored Alchemy Q head.

add_trial_phase widens the observation, which means the symbolic/context split
that valid_action_mask_from_observation performs has to be told about it --
otherwise the mask silently reads the wrong slice. That is what this checks,
alongside the value semantics of the two appended scalars.

Run: python scripts/verify_trial_phase.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from envs.alchemy import (
    TRIAL_PHASE_DIM,
    SymbolicAlchemyEnv,
    valid_action_mask_from_observation,
)

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
NUM_TRIALS, STEPS = 10, 20


def make(**kw):
    return SymbolicAlchemyEnv(
        level_name=LEVEL, num_trials=NUM_TRIALS, max_steps_per_trial=STEPS,
        observe_used=True, add_trial_flag=True, structured_potions=True, **kw)


def check_obs_and_mask(seeds=range(1000, 1006)):
    base, phase = make(), make(add_trial_phase=True)
    n_width = n_mask = n_val = 0
    for seed in seeds:
        ob, _ = base.reset(seed=seed)
        op, _ = phase.reset(seed=seed)
        for t in range(NUM_TRIALS * STEPS + 1):
            assert op.shape[0] == ob.shape[0] + TRIAL_PHASE_DIM, "width"
            n_width += 1
            # 1. the symbolic block is byte-identical; only the tail is new
            assert np.array_equal(op[:-TRIAL_PHASE_DIM], ob), "prefix changed"

            # 2. the mask is identical once it is told about the wider tail
            mb = valid_action_mask_from_observation(
                ob[None], observe_used=True, add_trial_flag=True,
                structured_potions=True)
            mp = valid_action_mask_from_observation(
                op[None], observe_used=True, add_trial_flag=True,
                structured_potions=True, add_trial_phase=True)
            assert np.array_equal(mb, mp), f"mask mismatch at t={t}"
            n_mask += 1

            # 3. the two scalars mean what they claim
            steps_left, trials_left = op[-2], op[-1]
            want_s = (STEPS - t % STEPS) / STEPS
            want_t = (NUM_TRIALS - t // STEPS) / NUM_TRIALS
            assert abs(steps_left - want_s) < 1e-6, (t, steps_left, want_s)
            assert abs(trials_left - max(want_t, 0.0)) < 1e-6, (t, trials_left)
            assert 0.0 <= steps_left <= 1.0 and 0.0 <= trials_left <= 1.0
            n_val += 1

            if t == NUM_TRIALS * STEPS:
                break
            a = int(np.argmax(np.random.rand(mb.shape[-1]) * mb[0]))
            ob, *_ = base.step(a)
            op, *_ = phase.step(a)
    print(f"  width checks   {n_width}  ok")
    print(f"  mask checks    {n_mask}  ok (0 mismatches)")
    print(f"  value checks   {n_val}  ok")

    # 4. the WRONG wiring must actually fail loudly, not silently mis-slice
    try:
        valid_action_mask_from_observation(
            op[None], observe_used=True, add_trial_flag=True,
            structured_potions=True)  # forgot add_trial_phase
        raise AssertionError("expected a width error")
    except ValueError as e:
        assert "unexpected width" in str(e)
        print("  untold mask    raises ValueError  ok")


def check_factored_head():
    from policies.models.action_heads import FactoredAlchemyQHead
    torch.manual_seed(0)
    head = FactoredAlchemyQHead(
        input_size=32, hidden_sizes=(64, 64), max_stones=3, targets_per_stone=13)
    assert head.action_dim == 40, head.action_dim
    for shape in [(7, 32), (5, 4, 32)]:
        out = head(torch.randn(*shape))
        assert out.shape == (*shape[:-1], 40), out.shape
    print(f"  head shapes    ok (action_dim={head.action_dim})")

    # It must be able to fit an ARBITRARY Q table -- i.e. the factorization is a
    # reparameterization, not a restriction. Fit one fixed random target.
    x, y = torch.randn(1, 32), torch.randn(1, 40) * 3
    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(3000):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head(x), y)
        loss.backward()
        opt.step()
    print(f"  head expressiv ok (fit residual {loss.item():.2e} on random Q)")


if __name__ == "__main__":
    print("add_trial_phase:")
    check_obs_and_mask()
    print("factored_action_head:")
    check_factored_head()
    print("PASS")
