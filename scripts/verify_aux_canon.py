"""A/B check for aux_canon_target: the auxiliary canonical-frame supervision.

`aux_canon_target` appends a 21-dim SUPERVISION TARGET to the Alchemy
observation (3 stones x 3 latent coords, then 12 potion latent type indices).
Two things have to hold and neither is visible from a training curve:

  1. the block is a LABEL, never an input -- the agent must excise it before
     RNN_head / the critic / the action mask see the observation; and
  2. every observation-width accounting site must be told about it, or the
     action mask silently reads the wrong slice.

Both are checked here against a lockstep `canonicalize_oracle=True` env (the
technique scripts/verify_canonicalize.py and scripts/probe_frame_map.py use),
so nothing in the observation pipeline is re-implemented.

Run: python scripts/verify_aux_canon.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ml_collections import ConfigDict  # noqa: E402

from envs.alchemy import (  # noqa: E402
    AUX_CANON_ABSENT,
    AUX_CANON_DIM,
    AUX_CANON_STONE_DIM,
    SymbolicAlchemyEnv,
    get_symbolic_alchemy_layout,
    valid_action_mask_from_observation,
)

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
NUM_TRIALS, STEPS = 10, 20
ENV_KW = dict(
    level_name=LEVEL, num_trials=NUM_TRIALS, max_steps_per_trial=STEPS,
    observe_used=True, add_trial_flag=True, structured_potions=True,
)
MASK_KW = dict(observe_used=True, add_trial_flag=True, structured_potions=True)

LAYOUT = get_symbolic_alchemy_layout(observe_used=True, structured_potions=True)
SF = LAYOUT.stone_feature_dim
PF = LAYOUT.potion_feature_dim
STONE_W = LAYOUT.max_stones * SF
SYM_DIM = LAYOUT.symbolic_obs_dim


def _canon_fields(sym):
    """(coords (3,3), types (12,)) read out of a canonicalized symbolic obs."""
    stones = sym[:STONE_W].reshape(LAYOUT.max_stones, SF)
    potions = sym[STONE_W:].reshape(LAYOUT.max_potions, PF)
    coords = stones[:, :3]
    types = np.argmax(potions[:, :3], axis=-1) * 2 + (potions[:, 3] > 0)
    return coords, types.astype(np.int64)


def _present(sym):
    stones = sym[:STONE_W].reshape(LAYOUT.max_stones, SF)
    potions = sym[STONE_W:].reshape(LAYOUT.max_potions, PF)
    return stones[:, -1] < 0.5, potions[:, -1] < 0.5


# ── 1/2/3. env-side: width, mask invariance, target correctness ────────────
def check_env(seeds=range(3000, 3006)):
    base = SymbolicAlchemyEnv(**ENV_KW)
    aux = SymbolicAlchemyEnv(**ENV_KW, aux_canon_target=True)
    canon = SymbolicAlchemyEnv(**ENV_KW, canonicalize_oracle=True)
    n_w = n_m = n_t = n_sent = 0
    rng = np.random.default_rng(0)

    for seed in seeds:
        ob, _ = base.reset(seed=seed)
        oa, _ = aux.reset(seed=seed)
        oc, _ = canon.reset(seed=seed)
        for t in range(NUM_TRIALS * STEPS + 1):
            # (a) width is exactly +21 and the prefix is byte-identical
            assert oa.shape[0] == ob.shape[0] + AUX_CANON_DIM, "width"
            assert np.array_equal(oa[:-AUX_CANON_DIM], ob), "prefix changed"
            n_w += 1

            # (b) the action mask is IDENTICAL with the flag on vs off
            mb = valid_action_mask_from_observation(ob[None], **MASK_KW)
            ma = valid_action_mask_from_observation(
                oa[None], aux_canon_target=True, **MASK_KW)
            assert np.array_equal(mb, ma), f"mask mismatch at t={t}"
            n_m += 1

            # (c) the targets equal what canonicalize_oracle=True produces
            tail = oa[-AUX_CANON_DIM:]
            want_c, want_t = _canon_fields(oc[:SYM_DIM])
            sm, pm = _present(ob[:SYM_DIM])
            got_c = tail[:AUX_CANON_STONE_DIM].reshape(LAYOUT.max_stones, 3)
            got_t = tail[AUX_CANON_STONE_DIM:]
            assert np.array_equal(got_c[sm], want_c[sm]), (t, got_c, want_c)
            assert np.array_equal(got_t[pm].astype(np.int64), want_t[pm]), t
            assert set(np.unique(got_c[sm]).tolist()) <= {-1.0, 1.0}
            n_t += 1

            # (d) absent slots carry the out-of-band sentinel, and only those
            assert np.all(got_c[~sm] == AUX_CANON_ABSENT)
            assert np.all(got_t[~pm] == AUX_CANON_ABSENT)
            assert np.all(got_c[sm] != AUX_CANON_ABSENT)
            assert np.all(got_t[pm] != AUX_CANON_ABSENT)
            n_sent += 1

            if t == NUM_TRIALS * STEPS:
                break
            valid = np.flatnonzero(mb[0])
            a = int(valid[rng.integers(len(valid))])
            ob, *_ = base.step(a)
            oa, *_ = aux.step(a)
            oc, *_ = canon.step(a)

    print(f"  width checks     {n_w}  ok (+{AUX_CANON_DIM})")
    print(f"  mask checks      {n_m}  ok (0 mismatches vs flag off)")
    print(f"  target checks    {n_t}  ok (== canonicalize_oracle lockstep)")
    print(f"  sentinel checks  {n_sent}  ok")

    # (e) the WRONG wiring must fail loudly, not silently mis-slice
    try:
        valid_action_mask_from_observation(oa[None], **MASK_KW)
        raise AssertionError("expected a width error")
    except ValueError as e:
        assert "unexpected width" in str(e)
        print("  untold mask      raises ValueError  ok")

    # (f) aux_canon_target + canonicalize_oracle must be refused
    try:
        SymbolicAlchemyEnv(**ENV_KW, aux_canon_target=True,
                           canonicalize_oracle=True)
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "canonicalize_oracle" in str(e)
        print("  canon+aux        raises ValueError  ok")


# ── agent-side ─────────────────────────────────────────────────────────────
def _configs(aux_target, aux_weight):
    import configs.rl.dqn_default as dqn_cfg
    import configs.seq_models.markov_default as seq_cfg

    config_rl = dqn_cfg.get_config()
    del config_rl.update_fn
    config_rl.init_eps, config_rl.end_eps, config_rl.schedule_steps = 1.0, 0.01, 1000
    config_rl.mask_alchemy_invalid_actions = True
    config_rl.use_popart = True
    config_rl.aux_canon_weight = aux_weight

    config_seq = seq_cfg.get_config()
    del config_seq.update_fn
    config_seq.seq_model.is_oracle = True
    config_seq.seq_model.context_dim = 28
    config_seq.seq_model.max_seq_length = 256
    config_seq.normalize_inputs = True
    config_seq.use_pe = True
    config_seq.conditioning_hidden_dim = 128
    config_seq.compile = False

    config_env = ConfigDict()
    config_env.env_type = "alchemy"
    config_env.observe_used = True
    config_env.add_trial_flag = True
    config_env.structured_potions = True
    config_env.structured_stones = False
    config_env.add_trial_phase = False
    config_env.aux_canon_target = aux_target
    return config_rl, config_seq, config_env


def _batch(T, B, obs_dim, action_dim, aux_obs_np, seed=0):
    g = torch.Generator().manual_seed(seed)
    actions = torch.nn.functional.one_hot(
        torch.randint(action_dim, (T + 1, B), generator=g), action_dim
    ).float()
    rewards = torch.randn(T + 1, B, 1, generator=g)
    terms = torch.zeros(T + 1, B, 1)
    masks = torch.ones(T + 1, B, 1)
    masks[-2:, B // 2:] = 0.0            # some padding, as in a real batch
    observs = torch.as_tensor(aux_obs_np, dtype=torch.float32)
    assert observs.shape == (T + 2, B, obs_dim), observs.shape
    return actions, rewards, observs, terms, masks


def _collect_obs(n_steps, B, aux_target, seed0=7000):
    """(n_steps, B, dim) of real observations from `B` lockstep episodes."""
    envs = [SymbolicAlchemyEnv(**ENV_KW, aux_canon_target=aux_target)
            for _ in range(B)]
    rng = np.random.default_rng(1)
    resets = [e.reset(seed=seed0 + i) for i, e in enumerate(envs)]
    obs = [r[0] for r in resets]
    ctxs = [r[1]["context"] for r in resets]  # chemistry: constant per episode
    out = []
    for _ in range(n_steps):
        out.append(np.stack([np.concatenate([o, c])
                             for o, c in zip(obs, ctxs)]))
        nxt = []
        for e, o in zip(envs, obs):
            m = valid_action_mask_from_observation(
                o[None], aux_canon_target=aux_target, **MASK_KW)[0]
            valid = np.flatnonzero(m)
            a = int(valid[rng.integers(len(valid))])
            nxt.append(e.step(a)[0])
        obs = nxt
    return np.stack(out).astype(np.float32)


def check_agent():
    import torchkit.pytorch_utils as ptu
    from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN

    ptu.set_gpu_mode(False)
    T, B = 12, 4
    obs_aux = _collect_obs(T + 2, B, aux_target=True)
    obs_plain = _collect_obs(T + 2, B, aux_target=False)
    action_dim = 40

    # the two collectors must agree except for the aux block
    layout_tail = SYM_DIM + 1
    keep = np.concatenate(
        [obs_aux[..., :layout_tail], obs_aux[..., layout_tail + AUX_CANON_DIM:]],
        axis=-1,
    )
    assert np.allclose(keep, obs_plain), "lockstep obs collectors diverged"

    # ---- A. target dims never reach the network -------------------------
    seen = {}
    orig_forward = None

    def run(aux_target, aux_weight, obs_np, seed=0, spy=False):
        torch.manual_seed(0)
        np.random.seed(0)
        c_rl, c_seq, c_env = _configs(aux_target, aux_weight)
        agent = ModelFreeOffPolicy_DQN_RNN(
            obs_dim=obs_np.shape[-1], action_dim=action_dim,
            config_seq=c_seq, config_rl=c_rl, config_env=c_env,
        )
        agent.train()
        if spy:
            head = agent.head
            real = head.forward

            def spy_forward(*a, **kw):
                seen["obs_dim"] = kw["observs"].shape[-1]
                seen["obs"] = kw["observs"].detach().clone()
                return real(*a, **kw)
            head.forward = spy_forward
        batch = _batch(T, B, obs_np.shape[-1], action_dim, obs_np, seed)
        loss, out = agent._compute_loss(*batch)
        agent.critic_optimizer.zero_grad()
        loss.backward()
        grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
                 for n, p in agent.named_parameters()}
        return agent, loss, out, grads

    agent_on, loss_on, out_on, _ = run(True, 1.0, obs_aux, spy=True)
    assert seen["obs_dim"] == obs_aux.shape[-1] - AUX_CANON_DIM, seen["obs_dim"]
    assert seen["obs_dim"] == agent_on.net_obs_dim
    exp = np.concatenate(
        [obs_aux[..., :layout_tail], obs_aux[..., layout_tail + AUX_CANON_DIM:]],
        axis=-1,
    )
    assert torch.allclose(seen["obs"], torch.as_tensor(exp)), "wrong slice"
    # and the label values are nowhere in what the network received
    assert not (seen["obs"] == AUX_CANON_ABSENT).any(), "sentinel leaked"
    print(f"  RNN_head input   {seen['obs_dim']} dims "
          f"(= raw {obs_aux.shape[-1]} - {AUX_CANON_DIM})  ok")
    print(f"  slice identity   ok (matches aux-off env byte for byte)")

    # act() and sample_random_action() must also strip
    ptu.device = torch.device("cpu")
    o = torch.as_tensor(obs_aux[0])
    seen.clear()
    step_seen = {}
    real_step = agent_on.head.step

    def spy_step(**kw):
        step_seen["obs"] = kw["obs"].shape[-1]
        step_seen["prev_obs"] = kw["prev_obs"].shape[-1]
        return real_step(**kw)
    agent_on.head.step = spy_step
    agent_on.eval()
    with torch.no_grad():
        agent_on.act(None, torch.zeros(B, action_dim), torch.zeros(B, 1), o, o,
                     deterministic=True, initial=True)
    assert step_seen["obs"] == agent_on.net_obs_dim, step_seen
    assert step_seen["prev_obs"] == agent_on.net_obs_dim, step_seen
    a = agent_on.sample_random_action(raw_obs=o)
    assert a.shape == (B, action_dim)
    print(f"  act()/random     both see {agent_on.net_obs_dim} dims  ok")

    # ---- B. metrics are sane and GPU-resident (no python scalars) --------
    for k in ("aux_canon_loss", "aux_canon_stone_loss", "aux_canon_potion_loss",
              "aux_canon_stone_acc", "aux_canon_potion_acc"):
        assert k in out_on, k
        assert torch.is_tensor(out_on[k]), k
        assert out_on[k].dim() == 0, k
    acc_s = out_on["aux_canon_stone_acc"]
    acc_p = out_on["aux_canon_potion_acc"]
    assert 0.0 <= float(acc_s) <= 1.0 and 0.0 <= float(acc_p) <= 1.0
    print(f"  metrics          aux_canon_loss={float(out_on['aux_canon_loss']):.4f} "
          f"stone_acc={float(acc_s):.3f} potion_acc={float(acc_p):.3f}  ok")

    # ---- C. weight 0.0 == pre-change path --------------------------------
    _, loss_off, out_off, g_off = run(True, 0.0, obs_aux)
    _, loss_ref, out_ref, g_ref = run(False, 0.0, obs_plain)
    assert torch.equal(loss_off, loss_ref), (float(loss_off), float(loss_ref))
    assert set(g_off) == set(g_ref), set(g_off) ^ set(g_ref)
    for n in g_off:
        if g_ref[n] is None:
            assert g_off[n] is None, n
            continue
        assert torch.equal(g_off[n], g_ref[n]), n
    assert not any(k.startswith("aux_canon") for k in out_off)
    print(f"  weight=0.0       loss {float(loss_off.detach()):.8f} == "
          f"{float(loss_ref.detach()):.8f}, all {len(g_off)} grads identical  ok")

    # ---- D. weight 1.0 actually changes the loss and reaches the trunk ---
    assert not torch.equal(loss_on, loss_off)
    _, _, _, g_on = run(True, 1.0, obs_aux)
    trunk = [n for n in g_on if n.startswith("head.") and g_on[n] is not None
             and g_ref.get(n) is not None]
    changed = [n for n in trunk if not torch.equal(g_on[n], g_ref[n])]
    assert changed, "aux gradient never reached the shared trunk"
    print(f"  weight=1.0       loss differs; {len(changed)}/{len(trunk)} "
          f"shared-trunk grads changed  ok")

    # ---- E. weight>0 without a target must be refused --------------------
    try:
        run(False, 1.0, obs_plain)
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        assert "aux_canon_target" in str(e)
        print("  weight w/o tgt   raises ValueError  ok")


if __name__ == "__main__":
    print("env side:")
    check_env()
    print("agent side:")
    check_agent()
    print("PASS")
