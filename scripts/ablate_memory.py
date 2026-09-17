"""Does the trained policy actually USE its memory?

The memory-content probe and the return disagree, and the disagreement is the
whole puzzle:

    GPT    memory-only CPC 0.876 (obs alone is 0.576)   return 156.1
    MATE   memory-only CPC 0.239                        return 160.9
    oracle chemistry sits in the OBSERVATION            return 243.6

GPT's memory demonstrably holds the episode's chemistry and GPT still finishes
below the no-chemistry floor of 173.5 -- lower, in fact, than MATE, whose memory
holds almost nothing. So "the memory is empty" cannot be the whole story.

This asks the question directly: replay the trained policy with its memory
read-out forced to zero. Everything else is untouched -- same weights, same
observation pathway, same positional encoding (PE is added AFTER the read-out,
so it survives and the conditioning signal degrades to exactly what markov
gets). Three conditions:

    normal     the policy as trained
    zeroed     memory read-out := 0
    shuffled   memory read-out taken from a DIFFERENT episode in the batch

If `zeroed` matches `normal`, the policy never used the memory and the whole
memory axis is downstream of a Q-function that learned to ignore that channel.
If `shuffled` matches `normal` but `zeroed` does not, the policy uses the
memory's magnitude but not its episode-specific content.

    python scripts/ablate_memory.py --checkpoint <path> [--episodes 64]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.envs.alchemy import LEVELS as _LEVELS
from envs.alchemy import SymbolicAlchemyEnv
from policies.models import AGENT_CLASSES


def make_env(ce):
    return SymbolicAlchemyEnv(
        level_name=_LEVELS[ce["env_name"]],
        num_trials=ce["num_trials"], max_steps_per_trial=ce["max_steps_per_trial"],
        observe_used=ce["observe_used"], add_trial_flag=ce["add_trial_flag"],
        structured_potions=ce.get("structured_potions", False),
        structured_stones=ce.get("structured_stones", False),
        aux_canon_target=ce.get("aux_canon_target", False),
        canonicalize_oracle=ce.get("canonicalize_oracle", False),
        context_graph_only=ce.get("context_graph_only", False),
        canon_potion_acc=ce.get("canon_potion_acc", 1.0),
        add_trial_phase=ce.get("add_trial_phase", False),
    )


def build_agent(ckpt):
    cfg = ckpt["config"]
    ce, cr, cs = cfg["config_env"], cfg["config_rl"], cfg["config_seq"]
    env = make_env(ce)
    obs_dim = env.observation_space.shape[0]
    if cs["seq_model"].get("is_oracle"):
        obs_dim += cs["seq_model"]["context_dim"]
    agent = AGENT_CLASSES[cr["algo"]](
        obs_dim=obs_dim, action_dim=env.action_space.n,
        config_seq=_dict_to_cfg(cs), config_rl=_dict_to_cfg(cr),
        config_env=_dict_to_cfg(ce),
    ).to(ptu.device)
    agent.load_training_state_dict(ckpt["agent_training_state"])
    agent.eval()
    return agent, env, ce


def _dict_to_cfg(d):
    from ml_collections import ConfigDict
    return ConfigDict(d)




@torch.no_grad()
def run(agent, ce, n_episodes, mode, seed0):
    """mode: normal | zeroed | shuffled. Returns per-episode returns."""
    envs = [make_env(ce) for _ in range(n_episodes)]
    head = agent.head
    real_forward = head.seq_model.forward

    def patched(inputs, h_0, **kw):
        out = real_forward(inputs, h_0, **kw)
        o = out[0]
        if mode == "zeroed":
            o = torch.zeros_like(o)
        elif mode == "shuffled" and o.shape[1] > 1:
            o = o[:, torch.randperm(o.shape[1], device=o.device)]
        return (o,) + tuple(out[1:])

    head.seq_model.forward = patched
    try:
        obs = np.stack([e.reset(seed=seed0 + i)[0] for i, e in enumerate(envs)])
        obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
        A = envs[0].action_space.n
        state = head.seq_model.get_zero_internal_state(batch_size=n_episodes)
        prev_a = torch.zeros((n_episodes, A), device=ptu.device)
        prev_r = torch.zeros((n_episodes, 1), device=ptu.device)
        prev_o = obs
        total = np.zeros(n_episodes)
        for t in range(envs[0].max_episode_steps):
            a, state, _ = agent.act(
                prev_internal_state=state, prev_action=prev_a, prev_reward=prev_r,
                prev_obs=prev_o, obs=obs, deterministic=True,
                initial=(t == 0), timestep=t,
            )
            idx = a.argmax(-1).cpu().numpy()
            nxt, rew = [], []
            for e, ai in zip(envs, idx):
                o2, r, term, trunc, _ = e.step(int(ai))
                nxt.append(o2); rew.append(r)
            total += np.asarray(rew)
            prev_o, prev_a = obs, a
            prev_r = torch.as_tensor(np.asarray(rew), dtype=torch.float32,
                                     device=ptu.device).unsqueeze(-1)
            obs = torch.as_tensor(np.stack(nxt), dtype=torch.float32, device=ptu.device)
    finally:
        head.seq_model.forward = real_forward
        for e in envs:
            e.close()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--seed", type=int, default=9000)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    agent, _, ce = build_agent(ck)
    name = ck.get("wandb_run_name", "?")
    model = ck["config"]["config_seq"]["seq_model"]["name"]
    print(f"{name}  ({model}, {ck['counters']['episodes']:,} episodes trained)")
    print(f"{a.episodes} eval episodes, deterministic, same seeds across conditions\n")
    print(f"{'condition':12s} {'return':>8s} {'sd':>7s}   {'vs normal':>10s}")
    print("-" * 46)
    base = None
    for mode in ("normal", "zeroed", "shuffled"):
        r = run(agent, ce, a.episodes, mode, a.seed)
        if base is None:
            base = r.mean()
        print(f"{mode:12s} {r.mean():8.1f} {r.std():7.1f}   {r.mean()-base:+10.1f}")


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "2")))
    main()
