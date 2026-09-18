"""Alchemy's own RL/seq defaults apply, and never beat an explicit flag.

configs/rl and configs/seq_models are shared by T-Maze, MuJoCo, Metaworld and
CARL, so Alchemy's tuned values (tau 0.003, use_pe, max_norm 0.2, PopArt,
lr 3e-5) live in configs/envs/alchemy.py instead of being raised globally. The
risk that buys is the opposite one: silently overriding a deliberate
--config_rl.tau on the command line, which would make every sweep a lie. Both
directions are checked here.

    python scripts/verify_alchemy_defaults.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from configs.envs import alchemy as alchemy_cfg
from configs.envs import mujoco as mujoco_cfg
from configs.rl import dqn_default
from configs.seq_models import mate_default
from utils.experiment import explicit_flags, finalize_training_configs

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


class FakeFlags:
    """Stand-in for absl FLAGS: the hook only ever setattr()s onto it."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build(env_module, argv, flags=None):
    ce = env_module.get_config()
    del ce.create_fn
    cr = dqn_default.get_config()
    cs = mate_default.get_config()
    sys.argv = ["main.py"] + argv
    out = finalize_training_configs(
        cr, cs, max_episode_steps=200, train_episodes=100, config_env=ce,
        flags=flags,
    )
    return out


EXPECT = {
    ("rl", "tau"): 0.003,
    ("rl", "critic_lr"): 3e-5,
    ("rl", "use_popart"): True,
    ("seq", "max_norm"): 0.2,
}
SHARED = {  # what the shared configs say, i.e. what other envs keep
    ("rl", "tau"): 0.001,
    ("rl", "critic_lr"): 1e-4,
    ("rl", "use_popart"): False,
    ("seq", "max_norm"): 0.1,
}


def main():
    # --- 1. plain Alchemy run picks up the tuned values --------------------
    cr, cs = build(alchemy_cfg, [])
    for (side, key), want in EXPECT.items():
        got = (cr if side == "rl" else cs)[key]
        check(f"default applied: config_{side}.{key}", got == want,
              f"{got} (shared default {SHARED[(side, key)]})")

    # --- 2. an explicit flag wins, in both spellings -----------------------
    # finalize does not parse argv into the config -- absl does that upstream --
    # so write the override in first, exactly as absl would, then confirm the
    # hook leaves it alone.
    ce = alchemy_cfg.get_config(); del ce.create_fn
    cr0, cs0 = dqn_default.get_config(), mate_default.get_config()
    cr0.tau, cs0.use_pe = 0.5, True           # what absl would have set
    sys.argv = ["main.py", "--config_rl.tau=0.5", "--config_seq.use_pe=True"]
    cr, cs = finalize_training_configs(
        cr0, cs0, max_episode_steps=200, train_episodes=100,
        config_env=ce, flags=None)
    check("explicit --config_rl.tau survives the hook", cr.tau == 0.5,
          f"tau {cr.tau}")
    check("explicit --config_seq.use_pe survives the hook (mate, would be False)",
          cs.use_pe is True, f"use_pe {cs.use_pe}")

    # --- use_pe is decided per MODEL, not per env --------------------------
    # markov has no memory read-out, so PE is its entire conditioning signal;
    # for a memory model the same flag adds an episode-invariant vector on top
    # of the only part that differs.
    from configs.seq_models import markov_default
    ce = alchemy_cfg.get_config(); del ce.create_fn
    sys.argv = ["main.py"]
    _, cs_m = finalize_training_configs(
        dqn_default.get_config(), markov_default.get_config(),
        max_episode_steps=200, train_episodes=100, config_env=ce, flags=None)
    check("use_pe defaults ON for markov (its only conditioning signal)",
          cs_m.use_pe is True)
    ce = alchemy_cfg.get_config(); del ce.create_fn
    _, cs_t = finalize_training_configs(
        dqn_default.get_config(), mate_default.get_config(),
        max_episode_steps=200, train_episodes=100, config_env=ce, flags=None)
    check("use_pe defaults OFF for a memory model", cs_t.use_pe is False)
    check("add_trial_phase defaults ON", ce.add_trial_phase is True)

    check("space-separated spelling is recognised",
          "config_rl.tau" in explicit_flags(
              ["main.py", "--config_rl.tau", "0.5"]))
    check("equals spelling is recognised",
          "config_rl.tau" in explicit_flags(
              ["main.py", "--config_rl.tau=0.5"]))
    check("top-level flags are tracked too (not just config.*)",
          explicit_flags(["main.py", "--updates_per_step=0.5"])
          == {"updates_per_step"})

    # --- 2b. top-level flag defaults ---------------------------------------
    f = FakeFlags(updates_per_step=0.1)
    build(alchemy_cfg, [], flags=f)
    check("default applied: --updates_per_step", f.updates_per_step == 0.025,
          f"{f.updates_per_step} (shared default 0.1)")
    f = FakeFlags(updates_per_step=0.5)
    build(alchemy_cfg, ["--updates_per_step=0.5"], flags=f)
    check("explicit --updates_per_step is not overwritten",
          f.updates_per_step == 0.5, f"stayed {f.updates_per_step}")
    f = FakeFlags(updates_per_step=0.1)
    build(mujoco_cfg, [], flags=f)
    check("mujoco keeps the shared --updates_per_step",
          f.updates_per_step == 0.1, f"{f.updates_per_step}")

    # --- 3. other environments are untouched -------------------------------
    cr, cs = build(mujoco_cfg, [])
    for (side, key), want in SHARED.items():
        got = (cr if side == "rl" else cs)[key]
        check(f"mujoco keeps the shared config_{side}.{key}", got == want, f"{got}")

    # --- 4. the hook is consumed, so it cannot fire twice ------------------
    ce = alchemy_cfg.get_config()
    del ce.create_fn
    sys.argv = ["main.py"]
    finalize_training_configs(
        dqn_default.get_config(), mate_default.get_config(),
        max_episode_steps=200, train_episodes=100, config_env=ce,
    )
    check("apply_defaults_fn is deleted after use",
          "apply_defaults_fn" not in ce)

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
