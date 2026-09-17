"""Can a MATE memory of a given width hold the oracle's context at all?

Before fixing baseline architectures by the rule "hidden_size =
conditioning_hidden_dim = h/2", the width has to be checked: if the memory
cannot represent the context even under direct supervision, no RL run at that
size can succeed and the sweep would measure the wrong thing.

This trains MATE's embedder END TO END on the context, with no RL:

    m_t = (init + sum_{i<=t} E_phi(x_i)) / (w + t)      ->  head  ->  c

`c` is the 28-dim chem_gt the oracle receives. On no_bottleneck only 12 of its
dims vary across episodes (dims 0-11 are the graph, constant when there is no
bottleneck), so those 12 binary dims are the target and chance is 0.5 per bit.

This is the most favourable case there is: exact labels at every step, no
credit assignment, no exploration. A width that fails here cannot work in RL.
Split is by EPISODE.

    python scripts/mate_capacity.py --episodes 600 --widths 64,128,256
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.envs.alchemy import LEVELS
from envs.alchemy import (
    SymbolicAlchemyEnv, valid_action_mask_from_observation,
)

ENV_KW = dict(num_trials=10, max_steps_per_trial=20, observe_used=True,
              add_trial_flag=True, structured_potions=True)


def collect(level, n_episodes, seed0):
    """(n_ep, T, trans_dim) transitions and (n_ep, 28) contexts."""
    env = SymbolicAlchemyEnv(level_name=LEVELS[level], **ENV_KW)
    A = env.action_space.n
    X, C = [], []
    for ep in range(n_episodes):
        rng = np.random.default_rng(seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        C.append(np.asarray(info["context"], dtype=np.float32))
        rows = []
        while True:
            m = np.asarray(valid_action_mask_from_observation(
                np.asarray(obs, dtype=np.float32), observe_used=True,
                add_trial_flag=True, structured_potions=True))
            a = int(rng.choice(np.flatnonzero(m)))
            nxt, r, term, trunc, _ = env.step(a)
            onehot = np.zeros(A, dtype=np.float32); onehot[a] = 1.0
            # the repo's transition tuple: (o_t, a_t, r_t, o_{t+1} - o_t)
            rows.append(np.concatenate(
                [obs, onehot, [r], np.asarray(nxt) - np.asarray(obs)]
            ).astype(np.float32))
            obs = nxt
            if term or trunc:
                break
        X.append(np.stack(rows))
    env.close()
    return np.stack(X), np.stack(C)


class MateProbe(nn.Module):
    """MATE's embedder + running mean + a linear read-out onto the context."""

    def __init__(self, in_dim, hidden, n_layer, n_out):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layer + 1):
            layers += [nn.Linear(d, hidden), nn.LeakyReLU()]
            d = hidden
        self.embedder = nn.Sequential(*layers)
        self.init_emb = nn.Parameter(torch.randn(hidden) * 0.01)
        self.log_w = nn.Parameter(torch.zeros(()))
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                      # x (B, T, D)
        z = self.embedder(x)                   # (B, T, H)
        w = self.log_w.exp()
        cum = z.cumsum(dim=1) + w * self.init_emb
        t = torch.arange(1, z.shape[1] + 1, device=x.device).view(1, -1, 1)
        return self.head(cum / (t + w))        # (B, T, n_out)


def run(width, n_layer, Xtr, Ytr, Xte, Yte, epochs, device, bs=32):
    model = MateProbe(Xtr.shape[-1], width, n_layer, Ytr.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best = 0.0
    n = Xtr.shape[0]
    for _ in range(epochs):
        model.train()
        for i in torch.randperm(n, device=device).split(bs):
            xb, yb = Xtr[i], Ytr[i].unsqueeze(1).expand(-1, Xtr.shape[1], -1)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            # read the memory at the LAST step: the whole episode has been seen
            logit = model(Xte)[:, -1, :]
            acc = ((logit > 0) == (Yte > 0.5)).float().mean().item()
        best = max(best, acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--widths", default="64,128,256")
    ap.add_argument("--n_layer", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--level", default="no_bottleneck")
    ap.add_argument("--seed", type=int, default=3300)
    a = ap.parse_args()

    print(f"collecting {a.episodes} episodes ({a.level}, masked-random) ...")
    X, C = collect(a.level, a.episodes, a.seed)
    varying = C.std(0) > 1e-6
    Y = C[:, varying]
    print(f"  transitions {X.shape}, context dims that vary: "
          f"{int(varying.sum())}/{C.shape[1]}")

    dev = ptu.device
    n_tr = int(0.8 * len(X))
    Xt = torch.as_tensor(X, device=dev)
    mu, sd = Xt[:n_tr].reshape(-1, Xt.shape[-1]).mean(0), \
             Xt[:n_tr].reshape(-1, Xt.shape[-1]).std(0) + 1e-6
    Xt = (Xt - mu) / sd
    Yt = torch.as_tensor(Y, device=dev)
    Xtr, Ytr, Xte, Yte = Xt[:n_tr], Yt[:n_tr], Xt[n_tr:], Yt[n_tr:]

    print(f"\n{'hidden_size':>12s} {'params':>10s} {'best test acc':>14s}  (chance 0.500)")
    print("-" * 56)
    for w in [int(v) for v in a.widths.split(",")]:
        torch.manual_seed(0)
        m = MateProbe(X.shape[-1], w, a.n_layer, Y.shape[1])
        acc = run(w, a.n_layer, Xtr, Ytr, Xte, Yte, a.epochs, dev)
        print(f"{w:12d} {sum(p.numel() for p in m.parameters()):10,d} {acc:14.3f}")


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "0")))
    main()
