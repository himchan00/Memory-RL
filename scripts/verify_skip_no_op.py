"""Does memory_skip_no_op actually remove NO_OP steps from MATE's mean?

The idea: a NO_OP transition says nothing about the chemistry, yet a uniform
mean gives it the same 1/200 weight as the transition that reveals a potion's
axis. Under the training action mask 66 of 200 steps are NO_OPs, so a third of
what MATE averages is "nothing happened". Excluding them zeroes the numerator
AND the denominator, so the memory is as if the step never occurred -- while
the transition stays in the buffer, because the critic still has to value it.

Three ways that can be silently wrong, each checked here:
  the weight reads the wrong slice   -- NO_OP is action 0 of a one-hot, and it
                                        must be read off the RAW action, never
                                        the InputNorm'd transition.
  the alignment is off by one        -- obs_shortcut drops the dummy row at
                                        t=-1 from the sequence inputs, so the
                                        weights have to lose the same row.
  train and rollout disagree         -- forward() batches, step() goes one at a
                                        time; if only one of them honours the
                                        flag the agent evaluates a memory it
                                        never trained.

    python scripts/verify_skip_no_op.py
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.envs import alchemy as env_cfg
from configs.rl import dqn_default
from configs.seq_models import mate_default
from envs.alchemy import NO_OP_ACTION, get_symbolic_alchemy_layout
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


def build(skip):
    ce = env_cfg.get_config(); del ce.create_fn
    ce.env_name = "no_bottleneck"; ce.structured_potions = True
    cr = dqn_default.get_config(); del cr.update_fn
    cr.init_eps, cr.end_eps, cr.schedule_steps = 0.0, 0.0, 1
    cs = mate_default.get_config(); del cs.update_fn
    cs.seq_model.max_seq_length = 64
    cs.seq_model.context_dim = 0
    cs.seq_model.is_oracle = False
    cs.seq_model.hidden_size = 64
    cs.seq_model.n_layer = 1
    cs.conditioning_hidden_dim = 32
    cs.conditioning_n_layer = 1
    cs.compile = False
    cs.memory_skip_no_op = skip
    layout = get_symbolic_alchemy_layout(True, True)
    obs_dim = layout.symbolic_obs_dim + 1
    agent = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=obs_dim, action_dim=40,
        config_seq=cs, config_rl=cr, config_env=ce,
    ).to(ptu.device)
    return agent.eval(), obs_dim


def main():
    T, B, A = 20, 3, 40
    torch.manual_seed(0)
    agent, obs_dim = build(skip=True)
    plain, _ = build(skip=False)
    plain.load_state_dict(agent.state_dict())
    plain.eval()

    # half the actions are NO_OP, in known places
    idx = torch.randint(1, A, (T + 1, B), device=ptu.device)
    no_op_rows = torch.zeros((T + 1, B), dtype=torch.bool, device=ptu.device)
    no_op_rows[::2] = True
    idx[no_op_rows] = NO_OP_ACTION
    actions = torch.nn.functional.one_hot(idx, A).float()
    rewards = torch.randn((T + 1, B, 1), device=ptu.device)
    observs = torch.randn((T + 1, B, obs_dim), device=ptu.device)
    next_observs = torch.randn((T + 1, B, obs_dim), device=ptu.device)
    masks = torch.ones((T + 1, B, 1), device=ptu.device)
    tt = torch.arange(T + 1, device=ptu.device).view(-1, 1).expand(T + 1, B)

    head = agent.head
    w = head._no_op_weights(actions)
    check("weight is 0 exactly on the NO_OP rows",
          bool((w.squeeze(-1) == 0).eq(no_op_rows).all()),
          f"{int((w == 0).sum())} zeroed of {w.numel()}")
    check("the plain head computes no weights at all",
          plain.head._no_op_weights(actions) is None)

    # --- the memory's effective count ---------------------------------------
    with torch.no_grad():
        (_, _, _), state_skip = _run(head, actions, rewards, observs,
                                     next_observs, masks, tt)
        (_, _, _), state_plain = _run(plain.head, actions, rewards, observs,
                                      next_observs, masks, tt)
    n_real = int((~no_op_rows[1:]).sum(0)[0])       # obs_shortcut drops row 0
    count_skip = float(state_skip[1][0, 0, 0])
    count_plain = float(state_plain[1][0, 0, 0])
    w0 = float(head.seq_model.get_zero_internal_state(batch_size=1)[1][0, 0, 0])
    check("count advances only on non-NO_OP steps",
          abs(count_skip - (w0 + n_real)) < 1e-4,
          f"count {count_skip:.3f}, expected {w0:.3f}+{n_real} ; "
          f"unskipped would be {count_plain:.3f}")
    check("skipping actually changes the memory",
          abs(count_skip - count_plain) > 1.0)

    # --- train vs rollout ----------------------------------------------------
    st = head.seq_model.get_zero_internal_state(batch_size=B)
    with torch.no_grad():
        for t in range(1, T + 1):
            _, st, _ = head.step(
                prev_internal_state=st,
                prev_action=actions[t:t + 1],
                prev_reward=rewards[t:t + 1],
                prev_obs=observs[t:t + 1],
                obs=next_observs[t:t + 1],
                initial=False, timestep=t,
            )
    d_sum = float((st[0] - state_skip[0]).abs().max())
    d_cnt = float((st[1] - state_skip[1]).abs().max())
    check("rollout step() matches the batched forward()",
          d_sum < 2e-4 and d_cnt < 1e-4,
          f"|d sum| {d_sum:.2e}  |d count| {d_cnt:.2e}")

    # --- guards --------------------------------------------------------------
    for name, mutate in (
        ("STORE", lambda cs: cs.seq_model.update(use_store=True)),
        ("a non-MATE seq model", lambda cs: cs.seq_model.update(name="lstm")),
    ):
        try:
            _build_bad(mutate); ok = False
        except ValueError:
            ok = True
        check(f"refused alongside {name}", ok)

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
        sys.exit(1)


def _run(head, actions, rewards, observs, next_observs, masks, tt):
    """forward(), plus the seq model's final internal state."""
    captured = {}
    real = head.seq_model.forward

    def spy(*a, **kw):
        out = real(*a, **kw)
        captured["state"] = out[1]
        return out
    head.seq_model.forward = spy
    try:
        out = head(actions, rewards, observs, next_observs, masks, tt)
    finally:
        head.seq_model.forward = real
    return out[:3], captured["state"]


def _build_bad(mutate):
    ce = env_cfg.get_config(); del ce.create_fn
    ce.env_name = "no_bottleneck"; ce.structured_potions = True
    cr = dqn_default.get_config(); del cr.update_fn
    cr.init_eps, cr.end_eps, cr.schedule_steps = 0.0, 0.0, 1
    cs = mate_default.get_config(); del cs.update_fn
    cs.seq_model.max_seq_length = 64
    cs.seq_model.context_dim = 0
    cs.seq_model.is_oracle = False
    cs.compile = False
    cs.memory_skip_no_op = True
    mutate(cs)
    layout = get_symbolic_alchemy_layout(True, True)
    ModelFreeOffPolicy_DQN_RNN(
        obs_dim=layout.symbolic_obs_dim + 1, action_dim=40,
        config_seq=cs, config_rl=cr, config_env=ce,
    )


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "3")))
    main()
