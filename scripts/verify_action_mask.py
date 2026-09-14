"""Check that the Alchemy action mask reaches every place it has to.

There are four, and missing any one of them is silent:

  act()             the greedy pick and the epsilon draw. A miss here wastes
                    steps on no-ops but is visible in the return.
  warm-up draw      Learner asks the agent for a uniform draw over LEGAL
                    actions; a miss fills the buffer with no-ops.
  Q-TARGET BOOTSTRAP  DDQN's argmax over the next state. A miss here is NOT
                    visible in the return until much later: an illegal action's
                    Q is never updated by experience, so it is unconstrained,
                    and a target that bootstraps through one grows without
                    bound. Measured on this task: q climbing 41 -> 1988 across
                    eight log points while the environment's own ceiling is 315,
                    with critic_loss and grad_norm both looking healthy.
  eval              the same path as act(), deterministic.

The third is the reason this file exists. It was the bug that made the first
port diverge, and nothing in the loss curves said so.

    python scripts/verify_action_mask.py
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
from configs.seq_models import markov_default
from envs.alchemy import SymbolicAlchemyEnv, get_symbolic_alchemy_layout
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN

T, B, ACTION_DIM = 16, 4, 40
PASS, FAIL = "  PASS", "  FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


ENV_KW = dict(
    level_name="perceptual_mapping_randomized",
    num_trials=2, max_steps_per_trial=8,
    observe_used=True, add_trial_flag=True, structured_potions=True,
)


def build(is_oracle):
    ce = env_cfg.get_config()
    del ce.create_fn
    ce.env_name = "no_bottleneck"
    ce.structured_potions = True
    cr = dqn_default.get_config()
    del cr.update_fn
    cr.init_eps, cr.end_eps, cr.schedule_steps = 1.0, 0.01, 1000
    cr.mask_alchemy_invalid_actions = True
    cr.use_popart = True
    cs = markov_default.get_config()
    del cs.update_fn
    cs.seq_model.max_seq_length = T + 2
    cs.seq_model.context_dim = 28 if is_oracle else 0
    cs.seq_model.is_oracle = is_oracle
    cs.conditioning_hidden_dim = 64
    cs.compile = False
    layout = get_symbolic_alchemy_layout(True, True)
    obs_dim = layout.symbolic_obs_dim + 1 + (28 if is_oracle else 0)
    agent = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=obs_dim, action_dim=ACTION_DIM,
        config_seq=cs, config_rl=cr, config_env=ce,
    ).to(ptu.device)
    return agent, obs_dim


def real_observations(n, B, is_oracle):
    """(n, B, dim) of REAL observations -- a synthetic tensor would give a mask
    that is all-true or all-false and the checks would pass vacuously."""
    envs = [SymbolicAlchemyEnv(**ENV_KW) for _ in range(B)]
    rng = np.random.default_rng(0)
    resets = [e.reset(seed=900 + i) for i, e in enumerate(envs)]
    obs = [r[0] for r in resets]
    ctxs = [r[1]["context"] for r in resets]
    out = []
    for _ in range(n):
        rows = [np.concatenate([o, c]) if is_oracle else o
                for o, c in zip(obs, ctxs)]
        out.append(np.stack(rows))
        nxt = []
        for e, o in zip(envs, obs):
            a = int(rng.integers(ACTION_DIM))
            s, _, term, trunc, _ = e.step(a)
            nxt.append(e.reset(seed=int(rng.integers(1 << 30)))[0]
                       if (term or trunc) else s)
        obs = nxt
    for e in envs:
        e.close()
    return np.stack(out).astype(np.float32)


def main():
    for is_oracle in (True, False):
        tag = "oracle" if is_oracle else "mate/gpt"
        print(f"\n=== {tag} ===")
        agent, obs_dim = build(is_oracle)
        agent.train()
        raw = torch.as_tensor(
            real_observations(T + 2, B, is_oracle), device=ptu.device
        )
        mask = agent.alchemy.valid_action_mask(raw)
        legal = mask.sum(-1).float()
        check("mask is non-trivial on real observations",
              bool((legal > 0).all() and (legal < ACTION_DIM).any()),
              f"legal per step: min {int(legal.min())} mean {float(legal.mean()):.1f} "
              f"max {int(legal.max())} of {ACTION_DIM}")

        # ---- the bootstrap target ----------------------------------------
        captured = {}
        real_argmax = torch.argmax

        def spy(x, *a, **kw):
            captured.setdefault("args", []).append(x.detach().clone())
            return real_argmax(x, *a, **kw)

        observs, next_observs = raw[:-1], raw[1:]
        actions = torch.nn.functional.one_hot(
            torch.randint(ACTION_DIM, (T + 1, B), device=ptu.device), ACTION_DIM
        ).float()
        rewards = torch.randn((T + 1, B, 1), device=ptu.device)
        terms = torch.zeros((T + 1, B, 1), device=ptu.device)
        masks = torch.ones((T + 1, B, 1), device=ptu.device)
        transition_t = torch.arange(T + 1, device=ptu.device).view(-1, 1).expand(T + 1, B)

        torch.argmax = spy
        try:
            agent._compute_loss(actions, rewards, observs, next_observs,
                                terms, masks, transition_t)
        finally:
            torch.argmax = real_argmax

        # the first argmax inside _compute_loss is the DDQN selection
        sel = captured["args"][0]
        next_mask = agent.alchemy.valid_action_mask(next_observs)
        chosen = sel.argmax(dim=-1, keepdim=True)
        legal_choice = next_mask.gather(-1, chosen)
        check("Q-target bootstrap never selects an illegal action",
              bool(legal_choice.all()),
              f"{int(legal_choice.sum())}/{legal_choice.numel()} legal")
        blocked = (sel <= -1e7)
        check("illegal entries are actually suppressed in the selection logits",
              bool((blocked == ~next_mask).all()),
              f"{int(blocked.sum())} suppressed, {int((~next_mask).sum())} illegal")

        # ---- rollout paths ------------------------------------------------
        agent.eval()
        bs = B
        internal = agent.head.seq_model.get_zero_internal_state(batch_size=bs)
        with torch.no_grad():
            a, _, _ = agent.act(
                prev_internal_state=internal,
                prev_action=torch.zeros((bs, ACTION_DIM), device=ptu.device),
                prev_reward=torch.zeros((bs, 1), device=ptu.device),
                prev_obs=raw[0], obs=raw[1],
                deterministic=True, initial=True,
            )
        m1 = agent.alchemy.valid_action_mask(raw[1])
        check("act(deterministic) returns a legal action",
              bool(m1.gather(-1, a.argmax(-1, keepdim=True)).all()))
        with torch.no_grad():
            a2, _, _ = agent.act(
                prev_internal_state=internal,
                prev_action=torch.zeros((bs, ACTION_DIM), device=ptu.device),
                prev_reward=torch.zeros((bs, 1), device=ptu.device),
                prev_obs=raw[0], obs=raw[1],
                deterministic=False, initial=True,
            )
        check("act(epsilon) returns a legal action",
              bool(m1.gather(-1, a2.argmax(-1, keepdim=True)).all()))
        rnd = agent.sample_random_action(raw_obs=raw[1])
        check("warm-up random draw is legal",
              bool(m1.gather(-1, rnd.argmax(-1, keepdim=True)).all()))

    print()
    if all(_results):
        print(f"ALL {len(_results)} CHECKS PASSED")
    else:
        print(f"{_results.count(False)} of {len(_results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "3")))
    main()
