"""Supervised probe: can our networks LEARN the perceived -> latent frame map?

Symbolic Alchemy RL plateaus at ~156 when the agent gets the observation in the
PERCEIVED (rotated/permuted) frame with the ground-truth chemistry `chem_gt`
concatenated on, but reaches 225+ when the env does the frame inversion for it
(`canonicalize_oracle=True`). The map IS a deterministic function of
(perceived obs, chem_gt[12:28]).  Open question: is it *representable /
optimizable* by a network of the size we actually use?

This script answers that with direct supervision, bypassing RL entirely.

  x = the 104-dim observation the agent actually sees
      = perceived symbolic_obs (75, structured_potions) ++ trial_flag (1)
        ++ chem_gt (28)
  y = the 75-dim symbolic_obs the SAME game state would have produced under
      `canonicalize_oracle=True`

The pair is produced by stepping TWO envs in lockstep on the same seed and the
same action sequence (the technique `scripts/verify_canonicalize.py` already
validates), so no part of the observation pipeline is re-implemented here.

Usage:
    conda run --no-capture-output -n mate-gpu \
        python scripts/probe_frame_map.py --episodes 1000 --epochs 50
"""
import argparse
import collections
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from envs.alchemy import (  # noqa: E402
    SymbolicAlchemyEnv,
    get_symbolic_alchemy_layout,
    valid_action_mask_from_observation,
)
from policies.models.conditioning import (  # noqa: E402
    ConcatConditioner,
    FiLMConditioner,
    HyperConditioner,
)

# ── layout constants (structured_potions=True, observe_used=True) ───────────
LAYOUT = get_symbolic_alchemy_layout(observe_used=True, structured_potions=True)
MAX_STONES = LAYOUT.max_stones                    # 3
MAX_POTIONS = LAYOUT.max_potions                  # 12
SF = LAYOUT.stone_feature_dim                     # 5 = [c0,c1,c2, reward/3, used]
PF = LAYOUT.potion_feature_dim                    # 5 = [ax0,ax1,ax2, dir, used]
STONE_W = MAX_STONES * SF                         # 15
SYM_DIM = LAYOUT.symbolic_obs_dim                 # 75
OBS_DIM = SYM_DIM + 1                             # 76 (+ trial flag)
CTX_DIM = 28
X_DIM = OBS_DIM + CTX_DIM                         # 104

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
ENV_KW = dict(
    level_name=LEVEL,
    num_trials=10,
    max_steps_per_trial=20,
    observe_used=True,
    add_trial_flag=True,
    structured_potions=True,
    structured_stones=False,
    add_trial_phase=False,
    context_graph_only=False,
)

# Fields the frame map is ALLOWED to touch: stone coords + potion axis/dir.
_ALLOWED = np.zeros(SYM_DIM, dtype=bool)
for _s in range(MAX_STONES):
    _ALLOWED[SF * _s: SF * _s + 3] = True
for _p in range(MAX_POTIONS):
    _ALLOWED[STONE_W + PF * _p: STONE_W + PF * _p + 4] = True


# ── field extraction ───────────────────────────────────────────────────────
def stone_coords(sym):
    """(N, 3, 3) coordinate triples."""
    return sym[:, :STONE_W].reshape(-1, MAX_STONES, SF)[:, :, :3]


def stone_present(sym):
    """(N, 3) bool — used flag < 0.5, exactly how the action mask reads it."""
    return sym[:, :STONE_W].reshape(-1, MAX_STONES, SF)[:, :, -1] < 0.5


def potion_block(sym):
    return sym[:, STONE_W:].reshape(-1, MAX_POTIONS, PF)


def potion_present(sym):
    return potion_block(sym)[:, :, -1] < 0.5


def potion_type(sym):
    """(N, 12) int in [0,6): axis*2 + (direction > 0). Garbage where absent."""
    blk = potion_block(sym)
    axis = np.argmax(blk[:, :, :3], axis=-1)
    return axis * 2 + (blk[:, :, 3] > 0).astype(np.int64)


# ── dataset ────────────────────────────────────────────────────────────────
def rollout_chunk(args):
    """One worker: `n_ep` lockstep episodes starting at `seed0`."""
    seed0, n_ep = args
    plain = SymbolicAlchemyEnv(**ENV_KW)
    canon = SymbolicAlchemyEnv(**ENV_KW, canonicalize_oracle=True)
    rng = np.random.default_rng(seed0)
    X, Y, EP = [], [], []
    for ep in range(n_ep):
        seed = seed0 + ep
        op, ip = plain.reset(seed=seed)
        oc, ic = canon.reset(seed=seed)
        assert np.array_equal(ip["context"], ic["context"]), "chemistry diverged"
        for _ in range(1000):
            X.append(np.concatenate([op, ip["context"]]))
            Y.append(oc[:SYM_DIM].copy())
            EP.append(seed)
            mask = valid_action_mask_from_observation(
                op[None], observe_used=True, add_trial_flag=True,
                context_dim=0, structured_potions=True)[0]
            valid = np.flatnonzero(mask)
            a = int(valid[rng.integers(len(valid))])
            op, _, _, tp, ip = plain.step(a)
            oc, _, _, tc, ic = canon.step(a)
            assert tp == tc and np.array_equal(ip["context"], ic["context"])
            if tp:
                break
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(EP, np.int64))


def build_dataset(n_episodes, workers, seed_base):
    per = max(1, n_episodes // workers)
    jobs = [(seed_base + i * per, per) for i in range(workers)]
    jobs[-1] = (jobs[-1][0], n_episodes - per * (workers - 1))
    if workers == 1:
        parts = [rollout_chunk(jobs[0])]
    else:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(rollout_chunk, jobs)
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    EP = np.concatenate([p[2] for p in parts])
    return X, Y, EP


# ── verification of the pairing ────────────────────────────────────────────
def verify(X, Y, EP, n_show=5):
    print("\n" + "=" * 78)
    print("PAIRING VERIFICATION")
    print("=" * 78)
    P = X[:, :SYM_DIM]                       # perceived symbolic part of x
    ctx = X[:, OBS_DIM:]
    ok = True

    # V1 — y may differ from the perceived obs ONLY in stone coords / potion type.
    diff = P != Y
    bad_cols = np.flatnonzero(diff.any(axis=0) & ~_ALLOWED)
    n_bad = int(diff[:, ~_ALLOWED].sum())
    print(f"V1  cols changed outside {{stone coords, potion axis/dir}}: "
          f"{len(bad_cols)} cols, {n_bad} elements   <- must be 0")
    ok &= n_bad == 0
    print(f"    (rewards / used flags / absent sentinels untouched; "
          f"{int(diff.any(axis=1).sum())}/{len(X)} steps differ at all)")

    # V2 — present-masks must agree in both frames (else the target is mis-masked).
    m1 = int((stone_present(P) != stone_present(Y)).sum())
    m2 = int((potion_present(P) != potion_present(Y)).sum())
    print(f"V2  present-mask mismatches: stones {m1}, potions {m2}   <- must be 0")
    ok &= (m1 == 0 and m2 == 0)

    # V3 — y must be a FUNCTION of x. Any two identical x with different y would
    #      mean the map is not computable from the agent's observation at all.
    seen, collisions, inconsistent = {}, 0, 0
    for xi, yi in zip(X, Y):
        k = xi.tobytes()
        if k in seen:
            collisions += 1
            if seen[k] != yi.tobytes():
                inconsistent += 1
        else:
            seen[k] = yi.tobytes()
    print(f"V3  duplicate-x rows: {collisions} ({len(seen)} unique x); "
          f"inconsistent y among them: {inconsistent}   <- must be 0")
    ok &= inconsistent == 0

    # V4 — the map must be a FUNCTION OF chem_gt[12:28] ALONE (plus the field
    #      being transformed).  This is the sharper version of "the identity
    #      frame leaves y == perceived": we recover the per-frame lookup table
    #      empirically and check it is single-valued and a bijection.
    #      Potions are keyed by (dim_map one-hot idx, dir_map bits) = dims 12-20,
    #      stones by (stone_map bits, rotation one-hot idx)          = dims 21-27.
    same = ~diff.any(axis=1)

    def _table(key, src, dst, mask, what, n_sym):
        """key/src/dst/mask are flat arrays over (sample x slot)."""
        k, s, d, m = key[mask], src[mask], dst[mask], mask.sum()
        rows = np.unique(np.stack([k, s, d], 1), axis=0)      # distinct (k,s,d)
        ks = np.unique(rows[:, :2], axis=0)                   # distinct (k,s)
        multi = len(rows) - len(ks)                           # >0 => not a function
        # bijection: within each key, #distinct src == #distinct dst
        keys, n_bad_bij, n_id = np.unique(k), 0, 0
        for kk in keys:
            r = rows[rows[:, 0] == kk]
            n_bad_bij += int(len(np.unique(r[:, 1])) != len(np.unique(r[:, 2])))
            n_id += int(np.array_equal(r[:, 1], r[:, 2]) and len(r) == len(
                np.unique(r[:, 1])))
        print(f"    {what}: {len(keys)} distinct frames, {m:,} labelled slots; "
              f"multi-valued (k,src) pairs: {multi}  <- must be 0; "
              f"non-bijective frames: {n_bad_bij}  <- must be 0")
        print(f"      identity frames observed: {n_id}"
              + (f" (of {n_sym} possible)" if n_sym else ""))
        return multi == 0 and n_bad_bij == 0

    N = len(X)
    # potions
    pkey = (np.argmax(ctx[:, 12:18], 1) * 8
            + (ctx[:, 18:21] > 0.5) @ np.array([4, 2, 1]))
    pk = np.repeat(pkey, MAX_POTIONS)
    ok &= _table(pk, potion_type(P).ravel(), potion_type(Y).ravel(),
                 potion_present(P).ravel(), "V4a potion type", 6 * 8)
    # stones: encode a coord triple in {-1,0,1}^3 as a base-3 int
    b3 = np.array([9, 3, 1])
    skey = ((ctx[:, 21:24] > 0.5) @ np.array([4, 2, 1]) * 4
            + np.argmax(ctx[:, 24:28], 1))
    sk = np.repeat(skey, MAX_STONES)
    src = ((stone_coords(P) + 1).astype(np.int64) @ b3).ravel()
    dst = ((stone_coords(Y) + 1).astype(np.int64) @ b3).ravel()
    ok &= _table(sk, src, dst, stone_present(P).ravel(), "V4b stone coords",
                 8 * 4)
    print(f"    => y is fully determined by (perceived obs, chem_gt[12:28]); "
          f"{int(same.sum())}/{N} steps happen to need no change at all")

    # V5 — chem_gt constant within an episode (sanity on the lockstep rollout).
    bad_ep = 0
    for e in np.unique(EP)[:50]:
        sel = ctx[EP == e]
        bad_ep += int(not np.all(sel == sel[0]))
    print(f"V5  episodes whose chem_gt is not constant: {bad_ep}/50   <- must be 0")
    ok &= bad_ep == 0

    # A couple of concrete samples.
    idx = np.flatnonzero(~same)[:2]
    for i in idx:
        print(f"\n    sample {i}: stone coords perceived -> latent")
        print("      ", stone_coords(P[i:i + 1])[0].tolist(), "->",
              stone_coords(Y[i:i + 1])[0].tolist(),
              " present:", stone_present(P[i:i + 1])[0].tolist())
        print("       potion types perceived -> latent")
        print("      ", potion_type(P[i:i + 1])[0].tolist(), "->",
              potion_type(Y[i:i + 1])[0].tolist())

    print("\nPAIRING VERIFICATION:", "PASS" if ok else "*** FAIL ***")
    assert ok, "pairing verification failed - the table below would be meaningless"
    return ok


# ── models ─────────────────────────────────────────────────────────────────
N_OUT = MAX_STONES * 3 + MAX_POTIONS * 6          # 9 + 72 = 81


class MLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU()]
            d = h
        layers += [nn.Linear(d, N_OUT)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CondHead(nn.Module):
    """Repo conditioner on (obs, chem_gt) + linear head. Mirrors RNN_head."""

    def __init__(self, cls, hidden_dim=128, n_layer=1, head_hidden=()):
        super().__init__()
        self.cond = cls(in_dim=OBS_DIM, out_dim=hidden_dim,
                        hidden_sizes=(hidden_dim,) * n_layer,
                        cond_dim=CTX_DIM, dropout=0.0)
        layers, d = [], self.cond.out_dim
        for h in head_hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU()]
            d = h
        layers += [nn.Linear(d, N_OUT)]
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.cond(x[:, :OBS_DIM], x[:, OBS_DIM:]))


# ── metrics ────────────────────────────────────────────────────────────────
def accuracy(pred_coord, pred_type, tgt_coord, tgt_type, sm, pm):
    """pred_coord (N,3,3) float; pred_type (N,12) int."""
    c_ok = ((pred_coord > 0) == (tgt_coord > 0))[sm].mean()
    p_ok = (pred_type == tgt_type)[pm].mean()
    return float(c_ok), float(p_ok)


def train_model(name, model, data, device, epochs, lr, bs, log):
    Xtr, Ytr_c, Ytr_t, smtr, pmtr, Xte, Yte_c, Yte_t, smte, pmte = data
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xtr)
    best = (0.0, 0.0, -1.0, 0)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            out = model(Xtr[j])
            pc = out[:, :9].reshape(-1, MAX_STONES, 3)
            pt = out[:, 9:].reshape(-1, MAX_POTIONS, 6)
            m = smtr[j]
            loss_c = (((pc - Ytr_c[j]) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
            mp = pmtr[j]
            ce = nn.functional.cross_entropy(
                pt.reshape(-1, 6), Ytr_t[j].reshape(-1), reduction="none"
            ).reshape(-1, MAX_POTIONS)
            loss_p = (ce * mp).sum() / mp.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            (loss_c + loss_p).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            outs = torch.cat([model(Xte[i:i + 8192])
                              for i in range(0, len(Xte), 8192)])
        pc = outs[:, :9].reshape(-1, MAX_STONES, 3)
        pt = outs[:, 9:].reshape(-1, MAX_POTIONS, 6).argmax(-1)
        c_ok, p_ok = accuracy(pc.cpu().numpy(), pt.cpu().numpy(),
                              Yte_c.cpu().numpy(), Yte_t.cpu().numpy(),
                              smte.cpu().numpy() > 0.5, pmte.cpu().numpy() > 0.5)
        score = c_ok + p_ok
        if score > best[2]:
            best = (c_ok, p_ok, score, ep)
        if log and (ep % 10 == 0 or ep == epochs - 1):
            print(f"    [{name}] epoch {ep:3d}  stone {c_ok:.4f}  potion {p_ok:.4f}")
    print(f"    [{name}] best @ epoch {best[3]}  ({time.time() - t0:.0f}s, "
          f"{sum(p.numel() for p in model.parameters()):,} params)")
    return best[0], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)   # x200 steps = 200k
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed_base", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    X, Y, EP = build_dataset(args.episodes, args.workers, args.seed_base)
    print(f"dataset: {X.shape[0]:,} steps from {len(np.unique(EP))} episodes "
          f"({time.time() - t0:.0f}s)   x {X.shape[1]}-dim, y {Y.shape[1]}-dim")

    verify(X, Y, EP)

    # ── split BY EPISODE (a chemistry never appears in both halves) ─────────
    eps = np.unique(EP)
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    test_eps = set(eps[: max(1, int(0.2 * len(eps)))].tolist())
    is_te = np.array([e in test_eps for e in EP])
    print(f"split: {int((~is_te).sum()):,} train / {int(is_te.sum()):,} test "
          f"steps, held out by episode ({len(test_eps)} test chemistries)")

    P = X[:, :SYM_DIM]
    Yc = stone_coords(Y)                        # (N,3,3) latent, in {-1,+1}
    Yt = potion_type(Y)                         # (N,12) latent type
    Pc = stone_coords(P)                        # perceived (identity baseline)
    Pt = potion_type(P)
    sm = stone_present(P).astype(np.float32)
    pm = potion_present(P).astype(np.float32)

    tr, te = ~is_te, is_te
    smte_b, pmte_b = sm[te] > 0.5, pm[te] > 0.5

    results = []

    # 1. identity — predict the perceived values unchanged.
    results.append(("identity (no training)",
                    *accuracy(Pc[te], Pt[te], Yc[te], Yt[te], smte_b, pmte_b)))

    # 2. majority — per-element train-set majority.
    maj_c = np.sign(np.where(sm[tr][:, :, None] > 0.5, Yc[tr], np.nan))
    maj_c = np.nan_to_num(np.nanmean(maj_c, axis=0))            # (3,3)
    maj_t = np.zeros(MAX_POTIONS, np.int64)
    for p in range(MAX_POTIONS):
        v = Yt[tr][:, p][pm[tr][:, p] > 0.5]
        maj_t[p] = np.bincount(v, minlength=6).argmax() if len(v) else 0
    results.append(("majority (no training)",
                    *accuracy(np.broadcast_to(maj_c, Yc[te].shape),
                              np.broadcast_to(maj_t, Yt[te].shape),
                              Yc[te], Yt[te], smte_b, pmte_b)))

    # ── tensors for the trained models ─────────────────────────────────────
    T = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=device)
    data = (T(X[tr]), T(Yc[tr]), T(Yt[tr], torch.long), T(sm[tr]), T(pm[tr]),
            T(X[te]), T(Yc[te]), T(Yt[te], torch.long), T(sm[te]), T(pm[te]))

    # MEMORYLESS CEILING. Same architecture, but the input is the perceived
    # observation ALONE -- no chem_gt. This is the bar an aux-head accuracy has
    # to clear before it counts as evidence that a MEMORY model inferred the
    # frame: anything at or below this line is readable from a single frame and
    # says nothing about what the memory holds.
    data_p = (T(P[tr]), T(Yc[tr]), T(Yt[tr], torch.long), T(sm[tr]), T(pm[tr]),
              T(P[te]), T(Yc[te]), T(Yt[te], torch.long), T(sm[te]), T(pm[te]))
    print("  training memoryless_256x2 (perceived obs only, no chem_gt) ...")
    c_ok, p_ok = train_model("memoryless_256x2", MLP(SYM_DIM, (256, 256)),
                             data_p, device, args.epochs, args.lr,
                             args.batch_size, args.verbose)
    results.append(("memoryless_256x2 (no chem_gt)  <- CEILING", c_ok, p_ok))

    zoo = [
        ("concat_256x2  <- our critic", MLP(X_DIM, (256, 256))),
        ("concat_512x4", MLP(X_DIM, (512, 512, 512, 512))),
        ("film   (h=128, n_layer=1)", CondHead(FiLMConditioner)),
        ("hypernet (h=128, n_layer=1)", CondHead(HyperConditioner)),
        ("repo path: concat-cond -> critic(256,256)",
         CondHead(ConcatConditioner, head_hidden=(256, 256))),
    ]
    for name, model in zoo:
        print(f"  training {name} ...")
        c_ok, p_ok = train_model(name, model, data, device,
                                 args.epochs, args.lr, args.batch_size,
                                 args.verbose)
        results.append((name, c_ok, p_ok))

    print("\n" + "=" * 78)
    print("SUPERVISED PROBE: perceived -> latent frame map (test set, held out "
          "by episode)")
    print("=" * 78)
    print(f"{'model':42s} {'stone-coord acc':>16s} {'potion-type acc':>16s}")
    print("-" * 78)
    for name, c, p in results:
        print(f"{name:42s} {c:16.4f} {p:16.4f}")
    print("-" * 78)
    print("chance: stone-coord 0.5 (sign), potion-type 0.1667 (6-way)")
    print(f"budget: Adam lr={args.lr}, batch={args.batch_size}, "
          f"{args.epochs} epochs, best-epoch test accuracy reported")


if __name__ == "__main__":
    main()
