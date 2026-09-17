"""How large is the PE next to the memory it is added to?

For a memory model the conditioning signal is

    c_t = m_t + pe_scale * PE(t)

a single 256-vector carrying both. PE(t) is IDENTICAL in every episode at the
same t; m_t is the only part that differs. If the PE term dominates, the
episode-specific part of c is a small perturbation on a large shared one -- and
the ablation on these same checkpoints found exactly that signature: zeroing
the memory costs 23 points, swapping in ANOTHER episode's memory costs nothing.

This measures the two norms on real rollouts, so the claim stops being an
argument about architecture and becomes a number.

    python scripts/pe_vs_memory.py --checkpoint <path>
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from scripts.ablate_memory import build_agent, make_env


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7700)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    agent, _, ce = build_agent(ck)
    head = agent.head
    if not head.use_pe:
        print("this checkpoint has use_pe=False; nothing to compare")
        return

    B = a.episodes
    envs = [make_env(ce) for _ in range(B)]
    obs = np.stack([e.reset(seed=a.seed + i)[0] for i, e in enumerate(envs)])
    obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
    A = envs[0].action_space.n
    state = head.seq_model.get_zero_internal_state(batch_size=B)
    prev_a = torch.zeros((B, A), device=ptu.device)
    prev_r = torch.zeros((B, 1), device=ptu.device)
    prev_o = obs

    mem_n, pe_n, across = [], [], []
    real = head.seq_model.forward

    def spy(inputs, h_0, **kw):
        out = real(inputs, h_0, **kw)
        h = out[0]
        if head.seq_model.name == "markov":
            h = h.new_zeros((h.shape[0], h.shape[1], head.cond_dim))
        m = h.squeeze(0)                                   # (B, D)
        mem_n.append(float(m.norm(dim=-1).mean()))
        # spread of the memory ACROSS episodes: this is the part PE cannot carry
        across.append(float((m - m.mean(0, keepdim=True)).norm(dim=-1).mean()))
        return out

    head.seq_model.forward = spy
    try:
        for t in range(envs[0].max_episode_steps):
            pe = head.pe_scale * head.pe(
                torch.as_tensor([t], device=ptu.device))
            pe_n.append(float(pe.norm()))
            act, state, _ = agent.act(
                prev_internal_state=state, prev_action=prev_a, prev_reward=prev_r,
                prev_obs=prev_o, obs=obs, deterministic=True,
                initial=(t == 0), timestep=t)
            idx = act.argmax(-1).cpu().numpy()
            nxt, rew = [], []
            for e, ai in zip(envs, idx):
                o2, r, term, trunc, _ = e.step(int(ai))
                nxt.append(o2); rew.append(r)
            prev_o, prev_a = obs, act
            prev_r = torch.as_tensor(np.asarray(rew), dtype=torch.float32,
                                     device=ptu.device).unsqueeze(-1)
            obs = torch.as_tensor(np.stack(nxt), dtype=torch.float32,
                                  device=ptu.device)
    finally:
        head.seq_model.forward = real
        for e in envs:
            e.close()

    mem = np.array(mem_n[1:]); pe = np.array(pe_n[1:len(mem)+1]); ac = np.array(across[1:])
    print(f"{ck.get('wandb_run_name')}  ({ck['config']['config_seq']['seq_model']['name']}, "
          f"{ck['counters']['episodes']:,} ep)")
    print(f"  pe_scale (learned)          {float(head.pe_scale):.4f}")
    print(f"  ||pe_scale * PE(t)||        {pe.mean():7.3f}")
    print(f"  ||m_t||                     {mem.mean():7.3f}")
    print(f"  ||m_t - mean_b(m_t)||       {ac.mean():7.3f}   <- the EPISODE-SPECIFIC part")
    print(f"\n  PE / memory                 {pe.mean()/max(mem.mean(),1e-9):7.2f}x")
    print(f"  PE / episode-specific part  {pe.mean()/max(ac.mean(),1e-9):7.2f}x")


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "2")))
    main()
