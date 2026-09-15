"""How much of the hidden context survives a running MEAN of transitions?

MATE's memory IS a running mean of learned transition embeddings:

    m_t = (init_emb + sum_i z_i) / (init_weight + t)

so whatever a mean of the transition tuples cannot separate, no amount of RL or
auxiliary supervision on top of MATE can recover. This script measures that
capacity directly, with NO training of the agent: roll a random policy, take the
running mean of the RAW transition tuples (o_t, a_t, r_t, o_{t+1}-o_t), and fit a
LINEAR probe from that mean to the episode's hidden context.

The linear probe is the charitable reading of MATE's embedder: `z = g(x)` can
rotate and rescale the input space, so anything linearly decodable from the mean
of x is reachable (a nonlinear g can do more, but this is the honest floor and it
is the same floor for both environments).

Two environments, same measurement:

  T-Maze passive   context = goal_y in {-1, +1}, one bit, revealed at t=0 only.
                   Chance 0.500. Swept over corridor length so the dilution
                   1/L is explicit.
  Alchemy          context = each potion slot's latent axis, 12 slots x 12
                   classes. Chance 0.083. Revealed only by the transition that
                   used that potion.

If dilution alone were the story, T-Maze at L=1000 (one informative step in a
thousand) would be harder than Alchemy at 200. It is not, and the reason this
script makes visible is that T-Maze's other 999 transitions are DETERMINISTIC:
they contribute the same vector in every episode, so they move the mean without
adding variance. Alchemy's other transitions are driven by which stone and
potion the policy happened to pick, which varies episode to episode and buries
the signal in interference rather than in a removable constant.

    python scripts/mean_information.py
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from envs.alchemy import SymbolicAlchemyEnv, get_symbolic_alchemy_layout
from envs.tmaze import TMazeClassicPassive


def transition_rows(obs_seq, act_seq, rew_seq, n_actions):
    """(T, D) of (o_t, onehot(a_t), r_t, o_{t+1} - o_t), the repo's tuple."""
    o = np.asarray(obs_seq, dtype=np.float64)
    a = np.zeros((len(act_seq), n_actions))
    a[np.arange(len(act_seq)), act_seq] = 1.0
    r = np.asarray(rew_seq, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([o[:-1], a, r, o[1:] - o[:-1]], axis=1)


def probe(X, y, n_train):
    """Held-out accuracy of a linear probe. Same split protocol everywhere."""
    if len(np.unique(y[:n_train])) < 2:
        return float("nan")
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((X[:n_train] - mu) / sd, y[:n_train])
    return float(clf.score((X[n_train:] - mu) / sd, y[n_train:]))


def run_tmaze(length, n_episodes, rng):
    env = TMazeClassicPassive(corridor_length=length, penalty=-1.0 / (length + 1))
    means, goals = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        seq_o, seq_a, seq_r = [obs], [], []
        goal = None
        for _ in range(length + 1):
            act = int(rng.integers(env.action_space.n))
            obs, rew, term, trunc, info = env.step(act)
            goal = int(info["context"][0])
            seq_o.append(obs); seq_a.append(act); seq_r.append(rew)
            if term or trunc:
                break
        rows = transition_rows(seq_o, seq_a, seq_r, env.action_space.n)
        means.append(rows.mean(axis=0))
        goals.append(goal)
    X, y = np.stack(means), np.array(goals)
    return probe(X, y, int(0.7 * len(y))), X.shape[1]


def run_alchemy(n_episodes, rng, n_trials=10, steps=20):
    layout = get_symbolic_alchemy_layout(True, True)
    env = SymbolicAlchemyEnv(
        level_name="perceptual_mapping_randomized", num_trials=n_trials,
        max_steps_per_trial=steps, observe_used=True, add_trial_flag=True,
        structured_potions=True, aux_canon_target=True,
    )
    n_act = env.action_space.n
    start = layout.symbolic_obs_dim + 1          # label block follows the trial flag
    means, labels = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
        # the 33-dim label block: 9 stone coords, then 12 potion slots
        potion = obs[start + 9: start + 21].copy()
        seq_o, seq_a, seq_r = [obs], [], []
        while True:
            act = int(rng.integers(n_act))
            obs, rew, term, trunc, _ = env.step(act)
            seq_o.append(obs); seq_a.append(act); seq_r.append(rew)
            if term or trunc:
                break
        rows = transition_rows(seq_o, seq_a, seq_r, n_act)
        means.append(rows.mean(axis=0))
        labels.append(potion)
    env.close()
    X, Y = np.stack(means), np.stack(labels)
    n_train = int(0.7 * len(X))
    accs = [probe(X, Y[:, s], n_train) for s in range(Y.shape[1])]
    accs = [a for a in accs if not np.isnan(a)]
    return float(np.mean(accs)), X.shape[1]


def main():
    rng = np.random.default_rng(0)
    N = 1200
    print(f"linear probe on the running MEAN of raw transition tuples, "
          f"{N} episodes, 70/30 split\n")
    print(f"{'env':26s} {'steps':>6s} {'informative':>12s} {'probe':>7s} "
          f"{'chance':>7s} {'lift':>7s}")
    print("-" * 70)
    for L in (10, 100, 1000):
        acc, _ = run_tmaze(L, N, rng)
        print(f"{'T-Maze passive L=' + str(L):26s} {L + 1:6d} "
              f"{'1 of ' + str(L + 1):>12s} {acc:7.3f} {0.5:7.3f} {acc - 0.5:+7.3f}")
    acc, _ = run_alchemy(N, rng)
    print(f"{'Alchemy no_bottleneck':26s} {200:6d} {'12 of 200':>12s} "
          f"{acc:7.3f} {1/12:7.3f} {acc - 1/12:+7.3f}")


if __name__ == "__main__":
    main()
