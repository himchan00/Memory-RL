"""Does a trained agent's memory hold the chemistry, when nothing asked it to?

The memory-content numbers quoted so far (GPT 0.876, MATE 0.239) come from runs
that carried aux_cpc at site="memory" -- a loss whose entire job is to put the
chemistry into the memory. The ablation that found the policy shuffle-invariant
was run on DIFFERENT checkpoints, trained with NO auxiliary loss at all. Whether
those memories hold anything was never measured, and the two claims cannot be
combined without it.

This probes a frozen checkpoint directly: roll the trained policy, record the
memory read-out m_t BEFORE the positional encoding is added, and fit a probe
from m_t to the episode's true potion frame map (which perceived potion slot
carries which latent axis and direction). Labels are read from the env's own
game state, so the agent's observation is untouched.

Split is by EPISODE, so a probe cannot win by memorising a trajectory. The
comparison point is the observation alone: if m_t does not beat that, the memory
is not adding anything the policy could have used.

    python scripts/probe_memory_content.py --checkpoint <path> [--episodes 120]
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from scripts.ablate_memory import build_agent, make_env

N_POTION_SLOTS = 12


def true_potion_types(env):
    """slot -> latent (axis, direction) for every potion still present."""
    gs = env._env.game_state
    out = {}
    for p in gs.existing_potions():
        out[gs.get_potion_ind(potion_inst=p.idx)] = (p.dimension, p.direction)
    return out


@torch.no_grad()
def collect(agent, ce, n_episodes, seed0, eps):
    """(memory, encoded_obs, label, episode_id) over rollouts."""
    head = agent.head
    envs = [make_env(ce) for _ in range(n_episodes)]
    obs = np.stack([e.reset(seed=seed0 + i)[0] for i, e in enumerate(envs)])
    obs = torch.as_tensor(obs, dtype=torch.float32, device=ptu.device)
    A = envs[0].action_space.n
    state = head.seq_model.get_zero_internal_state(batch_size=n_episodes)
    prev_a = torch.zeros((n_episodes, A), device=ptu.device)
    prev_r = torch.zeros((n_episodes, 1), device=ptu.device)
    prev_o = obs
    rng = np.random.default_rng(seed0)

    mem_rows, obs_rows, lab_rows, ep_rows = [], [], [], []
    real = head.seq_model.forward
    grabbed = {}

    def spy(inputs, h_0, **kw):
        out = real(inputs, h_0, **kw)
        h = out[0]
        if head.seq_model.name == "markov":
            h = h.new_zeros((h.shape[0], h.shape[1], head.cond_dim))
        grabbed["m"] = h.squeeze(0).detach().cpu().numpy()   # BEFORE the PE
        return out

    head.seq_model.forward = spy
    try:
        for t in range(envs[0].max_episode_steps):
            act, state, _ = agent.act(
                prev_internal_state=state, prev_action=prev_a, prev_reward=prev_r,
                prev_obs=prev_o, obs=obs, deterministic=True,
                initial=(t == 0), timestep=t)
            idx = act.argmax(-1).cpu().numpy()
            if eps > 0:                       # keep coverage off-policy
                flip = rng.random(n_episodes) < eps
                idx = np.where(flip, rng.integers(A, size=n_episodes), idx)
                act = torch.nn.functional.one_hot(
                    torch.as_tensor(idx, device=ptu.device), A).float()
            if t > 0 and "m" in grabbed:
                lab = np.full((n_episodes, N_POTION_SLOTS), -1, dtype=np.int64)
                for b, e in enumerate(envs):
                    for slot, (dim, d) in true_potion_types(e).items():
                        if slot < N_POTION_SLOTS:
                            lab[b, slot] = dim * 2 + (0 if d < 0 else 1)
                mem_rows.append(grabbed["m"].copy())
                obs_rows.append(obs.cpu().numpy().copy())
                lab_rows.append(lab)
                ep_rows.append(np.arange(n_episodes))
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
    return (np.concatenate(mem_rows), np.concatenate(obs_rows),
            np.concatenate(lab_rows), np.concatenate(ep_rows))


def probe(X, Y, ep, n_train_ep):
    tr, te = ep < n_train_ep, ep >= n_train_ep
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    accs = []
    for s in range(Y.shape[1]):
        ytr, yte = Y[tr, s], Y[te, s]
        ok_tr, ok_te = ytr >= 0, yte >= 0
        if ok_tr.sum() < 50 or ok_te.sum() < 20 or len(np.unique(ytr[ok_tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=600, C=1.0)
        clf.fit(Xtr[ok_tr], ytr[ok_tr])
        accs.append(clf.score(Xte[ok_te], yte[ok_te]))
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=5500)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    agent, _, ce = build_agent(ck)
    mem, obs, lab, ep = collect(agent, ce, a.episodes, a.seed, a.eps)
    n_tr = int(0.7 * a.episodes)
    name = ck.get("wandb_run_name")
    model = ck["config"]["config_seq"]["seq_model"]["name"]
    aux = (ck["config"]["config_rl"].get("aux_cpc_weight") or 0,
           ck["config"]["config_rl"].get("aux_canon_weight") or 0)
    print(f"{name}  ({model}, {ck['counters']['episodes']:,} ep, "
          f"aux_cpc={aux[0]} aux_canon={aux[1]})")
    print(f"{len(mem):,} rows, split by episode {n_tr}/{a.episodes - n_tr}\n")
    print(f"  probe on MEMORY m_t          {probe(mem, lab, ep, n_tr):.3f}")
    print(f"  probe on OBSERVATION o_t     {probe(obs, lab, ep, n_tr):.3f}")
    print(f"  probe on BOTH                {probe(np.concatenate([mem, obs], 1), lab, ep, n_tr):.3f}")
    print(f"  chance (6-way)               {1/6:.3f}")


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "2")))
    main()
