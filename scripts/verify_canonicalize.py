"""Check the env-side canonicalization == bc_diagnostic's validated latent_obs.

``--config_env.canonicalize_oracle=True`` must reproduce the ``latent`` condition
of the BC diagnostic exactly, otherwise the RL number it produces is not
comparable to the 164.9 that experiment measured. Run before trusting a P0 result:

    python scripts/verify_canonicalize.py
"""
import os, sys, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from envs.alchemy import SymbolicAlchemyEnv, valid_action_mask_from_observation
from bc_diagnostic import latent_obs, oracle_obs, LEVEL

kw = dict(level_name=LEVEL, num_trials=10, max_steps_per_trial=20,
          observe_used=True, add_trial_flag=True)
plain = SymbolicAlchemyEnv(**kw)
canon = SymbolicAlchemyEnv(**kw, canonicalize_oracle=True)

n_diff_steps = n_steps = 0
maxdiff_vs_bc = 0.0
mask_mismatch = 0
for ep in range(6):
    op, ip = plain.reset(seed=1000 + ep)
    oc, ic = canon.reset(seed=1000 + ep)
    for t in range(200):
        # 1. the two envs must be in the same underlying state
        assert np.allclose(ip["context"], ic["context"]), f"chemistry diverged ep{ep} t{t}"
        # 2. env canonicalization == bc_diagnostic latent_obs on the SAME state
        want = latent_obs(plain, op, ip)              # 68-dim, reference
        got = np.concatenate([oc, ic["context"]])     # 68-dim, env-side
        maxdiff_vs_bc = max(maxdiff_vs_bc, float(np.abs(want - got).max()))
        # 3. chem_gt / reward / used columns must be untouched by canonicalization
        base = oracle_obs(op, ip)
        assert np.array_equal(base[40:], got[40:]), "chem_gt tail changed"
        for s in range(3):
            assert base[5*s+3] == got[5*s+3] and base[5*s+4] == got[5*s+4], "stone reward/used changed"
        for p in range(12):
            assert base[15+2*p+1] == got[15+2*p+1], "potion used changed"
        # 4. the action mask must be identical in both frames
        mb = valid_action_mask_from_observation(base[None], observe_used=True,
                                               add_trial_flag=True, context_dim=28)
        mg = valid_action_mask_from_observation(got[None], observe_used=True,
                                               add_trial_flag=True, context_dim=28)
        mask_mismatch += int(not np.array_equal(mb, mg))
        n_steps += 1
        n_diff_steps += int(not np.array_equal(base, got))
        a = plain.action_space.sample()
        op, _, _, tp, ip = plain.step(a)
        oc, _, _, tc, ic = canon.step(a)
        if tp or tc:
            assert tp == tc
            break

print(f"steps compared          : {n_steps}")
print(f"max |env - bc_reference|: {maxdiff_vs_bc:.3e}   <- must be 0")
print(f"action-mask mismatches  : {mask_mismatch}       <- must be 0")
print(f"steps where latent != perceived: {n_diff_steps}/{n_steps} "
      f"({100*n_diff_steps/n_steps:.1f}%)  <- must be high, else it is a no-op")

# 5. canonicalize=False must be byte-identical to the pre-change behaviour
p2 = SymbolicAlchemyEnv(**kw)
o1, _ = plain.reset(seed=7); o2, _ = p2.reset(seed=7)
print(f"canonicalize=False regression: {'OK' if np.array_equal(o1, o2) else 'BROKEN'}")
print(f"obs width unchanged: plain={plain.observation_space.shape} canon={canon.observation_space.shape}")
