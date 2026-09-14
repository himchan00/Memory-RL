"""Does MATE's memory accumulate chemistry WITHIN an episode -- and if not, why?

`mate_probe_ctrl` measured per-trial potion accuracy under the TRAINED policy
and found it flat-to-falling (t0 .244 -> t9 .208). Two explanations fit:

  exploration collapse -- the memory stores fine, but the converged policy stops
                          running informative experiments;
  dilution             -- the memory cannot accumulate, because the transition
                          it averages is 193 dims of which ~14 carry the
                          experiment's result.

This separates them by holding the MEMORY FIXED (one checkpoint) and swapping
only the DATA: trajectories from the trained policy vs from a uniform-random
policy. A fresh probe is fit offline on each, identically.

  rising on random, flat on trained -> exploration collapse
  flat on both                      -> dilution

Each condition also fits an obs-only probe (memory zeroed) as an internal
control, so "what the memory adds" is read as a gap rather than an absolute.

    python scripts/probe_memory_accumulation.py \
        --run logs/count/alchemy/*/mate_probe_ctrl_*/
"""
import argparse, glob, os, sys
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from ml_collections import ConfigDict
from envs.alchemy import (
    AUX_CANON_NUM_POTION_TYPES,
    AUX_CANON_POTION_DIM,
    AUX_CANON_STONE_DIM,
    SymbolicAlchemyEnv,
    present_flags_from_observation,
)
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN
from torchkit.networks import FlattenMlp
from utils.checkpointing import load_training_checkpoint

LEVELS = {
    "rotation_random_bottleneck":
        "perceptual_mapping_randomized_with_rotation_and_random_bottleneck",
}


def build_agent(run_dir):
    ckpt = load_training_checkpoint(
        os.path.join(run_dir, "training_checkpoint.pth"), map_location="cpu"
    )
    cfg = ckpt["config"]
    ce, cr, cs = (ConfigDict(cfg[k]) for k in ("config_env", "config_rl", "config_seq"))
    env = SymbolicAlchemyEnv(
        level_name=LEVELS[ce.env_name],
        num_trials=ce.num_trials,
        max_steps_per_trial=ce.max_steps_per_trial,
        observe_used=ce.observe_used,
        add_trial_flag=ce.add_trial_flag,
        structured_potions=ce.structured_potions,
        aux_canon_target=ce.aux_canon_target,
    )
    obs_dim = env.observation_space.shape[0]
    agent = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=obs_dim, action_dim=env.action_space.n,
        config_seq=cs, config_rl=cr, config_env=ce, freeze_critic=False,
    ).to(ptu.device)
    agent.load_state_dict(ckpt["agent_training_state"]["model"])
    agent.eval()
    print(f"loaded {run_dir}  (env_steps {ckpt['counters']['env_steps']:,})")
    return agent, ce, obs_dim


def rollout(agent, ce, obs_dim, n_episodes, batch, random_policy, seed0):
    """-> observs (T+2,B,obs), actions (T+1,B,A), rewards (T+1,B,1)."""
    T = ce.num_trials * ce.max_steps_per_trial
    A = agent.action_dim
    obs_all, act_all, rew_all = [], [], []

    for start in range(0, n_episodes, batch):
        B = min(batch, n_episodes - start)
        envs = [
            SymbolicAlchemyEnv(
                level_name=LEVELS[ce.env_name], num_trials=ce.num_trials,
                max_steps_per_trial=ce.max_steps_per_trial,
                observe_used=ce.observe_used, add_trial_flag=ce.add_trial_flag,
                structured_potions=ce.structured_potions,
                aux_canon_target=ce.aux_canon_target,
            ) for _ in range(B)
        ]
        obs = np.stack([e.reset(seed=seed0 + start + i)[0] for i, e in enumerate(envs)])
        # buffer layout: a dummy step at t=-1, so observs is (T+2, B, ·)
        observs = torch.zeros((T + 2, B, obs_dim), device=ptu.device)
        actions = torch.zeros((T + 1, B, A), device=ptu.device)
        rewards = torch.zeros((T + 1, B, 1), device=ptu.device)
        observs[1] = ptu.from_numpy(obs)

        # RNN_head.step builds the zero state itself when initial=True, exactly
        # as Learner.collect_rollouts does.
        internal = None
        prev_a = torch.zeros((B, A), device=ptu.device)
        prev_r = torch.zeros((B, 1), device=ptu.device)
        prev_o = ptu.from_numpy(obs)

        for t in range(T):
            cur = ptu.from_numpy(obs)
            with torch.no_grad():
                a, internal = agent.act(
                    prev_internal_state=internal, prev_action=prev_a,
                    prev_reward=prev_r, prev_obs=prev_o, obs=cur,
                    deterministic=True, initial=(t == 0), timestep=t,
                )
                if random_policy:
                    # RAW obs: sample_random_action strips the aux block itself.
                    # Masked, so this is "uniform over LEGAL actions" -- the same
                    # distribution the agent's own epsilon-exploration draws from,
                    # not the uniform-over-40 floor.
                    a = agent.sample_random_action(raw_obs=cur)
            idx = a.argmax(dim=-1).cpu().numpy()
            nxt, rew = [], []
            for i, e in enumerate(envs):
                o2, r, term, trunc, _ = e.step(int(idx[i]))
                nxt.append(o2); rew.append(r)
            obs = np.stack(nxt)
            r_t = ptu.from_numpy(np.asarray(rew, dtype=np.float32)).reshape(B, 1)
            actions[t + 1] = a
            rewards[t + 1] = r_t
            observs[t + 2] = ptu.from_numpy(obs)
            prev_a, prev_r, prev_o = a, r_t, cur

        obs_all.append(observs); act_all.append(actions); rew_all.append(rewards)
        print(f"    rolled {start + B}/{n_episodes}", flush=True)

    return (torch.cat(obs_all, 1), torch.cat(act_all, 1), torch.cat(rew_all, 1))


@torch.no_grad()
def memory_readouts(agent, observs, actions, rewards):
    """-> encoded_obs, memory (both (T+2, B, ·), detached), and the labels."""
    agent.head.expose_memory_embeds = True
    stripped = agent._strip_aux_target(observs)
    masks = torch.ones((actions.shape[0], actions.shape[1], 1), device=ptu.device)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=stripped, masks=masks
    )
    targets = agent._aux_target_slice(observs)
    return d["_encoded_obs"], d["_memory_embeds"], targets, stripped


def trial_index_control(mem, n_trials, trial_len, steps=2000):
    """POSITIVE CONTROL: decode the trial index from the memory alone.

    A running mean over transitions must encode elapsed time -- the count is
    literally part of MATE's internal state. If a probe cannot recover even
    this from the memory, the probe (not the memory) is what is broken, and no
    conclusion may be drawn from the chemistry numbers. Held-out by episode,
    same protocol as the chemistry probe.
    """
    m = mem[1:-1]                                   # (T, B, ·), env steps 0..T-1
    T, B, D = m.shape
    y = (torch.arange(T, device=m.device) // trial_len).view(T, 1).expand(T, B)
    n_train = max(1, int(B * 0.75))
    with torch.no_grad():
        flat = m[:, :n_train].reshape(-1, D)
        mu, sigma = flat.mean(0), flat.std(0).clamp(min=1e-6)
    m = (m - mu) / sigma

    head = FlattenMlp(input_size=D, output_size=n_trials, hidden_sizes=(256, 256)).to(ptu.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(steps):
        cols = torch.randint(0, n_train, (min(64, n_train),), device=ptu.device)
        loss = F.cross_entropy(
            head(m[:, cols]).reshape(-1, n_trials), y[:, cols].reshape(-1)
        )
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(m[:, n_train:]).argmax(-1)
        return (pred == y[:, n_train:]).float().mean().item()


def fit_probe(agent, enc, mem, targets, stripped, use_memory, steps, n_trials, trial_len):
    """Fit a fresh potion-type probe; return (overall_acc, per_trial_acc)."""
    x = torch.cat((enc, mem if use_memory else torch.zeros_like(mem)), dim=-1)[:-1]
    # Bounded slice: the target block also carries the graph now, so an
    # open-ended [STONE_DIM:] would sweep it in.
    tgt = targets[
        :-1, ..., AUX_CANON_STONE_DIM:AUX_CANON_STONE_DIM + AUX_CANON_POTION_DIM
    ].long().clamp(
        0, AUX_CANON_NUM_POTION_TYPES - 1
    )
    _, potion_present = present_flags_from_observation(
        stripped[:-1], **agent._alchemy_split_kwargs
    )
    mask = potion_present.float()

    # Held-out episodes: the probe must generalise to chemistries it never fit,
    # or a large MLP could memorise per-episode maps and report a flat curve for
    # the wrong reason.
    B = x.shape[1]
    n_train = max(1, int(B * 0.75))

    # Standardize per feature on the TRAIN episodes. This matters more than it
    # usually would: the memory readout carries a large near-constant component
    # (norm ~2.5) with only ~1% chemistry-dependent variation, so an unscaled
    # probe is asked to find a needle sitting on a mountain. Without this, a
    # poorly conditioned input is indistinguishable from an uninformative one.
    with torch.no_grad():
        flat = x[:, :n_train].reshape(-1, x.shape[-1])
        mu, sigma = flat.mean(0), flat.std(0).clamp(min=1e-6)
    x = (x - mu) / sigma

    head = FlattenMlp(
        input_size=x.shape[-1],
        output_size=tgt.shape[-1] * AUX_CANON_NUM_POTION_TYPES,
        hidden_sizes=(256, 256),
    ).to(ptu.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)

    train_cols = torch.arange(n_train, device=ptu.device)
    test_cols = torch.arange(n_train, B, device=ptu.device)
    if len(test_cols) == 0:
        test_cols = train_cols

    def held_out_acc():
        with torch.no_grad():
            xe, te, me = x[:, test_cols], tgt[:, test_cols], mask[:, test_cols]
            hit = (
                head(xe).reshape(*te.shape, AUX_CANON_NUM_POTION_TYPES).argmax(-1) == te
            ).float()
            return ((hit * me).sum() / me.sum().clamp(min=1.0)).item()

    # EARLY STOPPING on held-out. Without it this probe reaches in-sample 1.000
    # -- it identifies the episode from the observation and memorises that
    # episode's map -- and a catastrophically overfit probe's final held-out
    # score underestimates what a well-regularised one could extract. Taking the
    # BEST held-out score over the fit makes the reported number an upper bound
    # on the decodable signal rather than an artefact of overfitting.
    best_acc, best_state, best_step = -1.0, None, 0
    for step in range(steps):
        cols = train_cols[torch.randint(0, n_train, (min(64, n_train),), device=ptu.device)]
        xb, tb, mb = x[:, cols], tgt[:, cols], mask[:, cols]
        logits = head(xb).reshape(*tb.shape, AUX_CANON_NUM_POTION_TYPES)
        ce = F.cross_entropy(
            logits.reshape(-1, AUX_CANON_NUM_POTION_TYPES), tb.reshape(-1),
            reduction="none",
        ).reshape(tb.shape)
        loss = (ce * mb).sum() / mb.sum().clamp(min=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0 or step == steps - 1:
            acc = held_out_acc()
            if acc > best_acc:
                best_acc, best_step = acc, step
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    if best_state is not None:
        head.load_state_dict(best_state)
    print(f"      early stop @ step {best_step} (best held-out {best_acc:.3f})")

    def evaluate(cols):
        xe, te, me = x[:, cols], tgt[:, cols], mask[:, cols]
        logits = head(xe).reshape(*te.shape, AUX_CANON_NUM_POTION_TYPES)
        hit = (logits.argmax(-1) == te).float()
        overall = ((hit * me).sum() / me.sum().clamp(min=1.0)).item()
        # row 0 is the dummy step at t=-1; rows 1..T are env steps 0..T-1
        h = (hit * me)[1:].reshape(n_trials, trial_len, *hit.shape[1:])
        m = me[1:].reshape(n_trials, trial_len, *me.shape[1:])
        per_trial = (h.flatten(1).sum(-1) / m.flatten(1).sum(-1).clamp(min=1.0))
        return overall, per_trial.cpu().numpy()

    with torch.no_grad():
        # IN-SAMPLE is reported too, because the in-run `aux_canon_potion_acc`
        # is an in-sample number: that head is scored on the batch it just
        # trained on, and over 480k updates it sees each buffered episode ~1000
        # times. If in-sample >> held-out here, the run's 0.553/0.222 figures
        # are substantially memorised per-episode chemistry, not a general
        # perceived->latent map read out of the memory.
        train_acc, train_curve = evaluate(train_cols)
        test_acc, test_curve = evaluate(test_cols)
    return (test_acc, test_curve), (train_acc, train_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir (glob ok)")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--probe_steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--cache", default="", help="dir to cache rollouts in")
    # GPU 3 is the only card we own on this box; 0-2 are other users'.
    ap.add_argument("--device", type=int, default=3)
    args = ap.parse_args()

    ptu.set_gpu_mode(torch.cuda.is_available(), args.device)
    run_dir = sorted(glob.glob(args.run))[0].rstrip("/")
    agent, ce, obs_dim = build_agent(run_dir)
    n_trials, trial_len = ce.num_trials, ce.max_steps_per_trial

    results = {}
    for label, random_policy in (("trained", False), ("random", True)):
        cache = (
            os.path.join(args.cache, f"rollout_{label}_{args.episodes}.pt")
            if args.cache else ""
        )
        if cache and os.path.exists(cache):
            print(f"\n[{label} policy] loading cached rollout {cache}")
            blob = torch.load(cache, map_location=ptu.device)
            observs, actions, rewards = blob["observs"], blob["actions"], blob["rewards"]
        else:
            print(f"\n[{label} policy] rolling out {args.episodes} episodes...")
            observs, actions, rewards = rollout(
                agent, ce, obs_dim, args.episodes, args.batch, random_policy,
                args.seed + (0 if random_policy else 100000),
            )
            if cache:
                os.makedirs(args.cache, exist_ok=True)
                torch.save(
                    {"observs": observs, "actions": actions, "rewards": rewards},
                    cache,
                )
        enc, mem, targets, stripped = memory_readouts(agent, observs, actions, rewards)
        # Sanity: a memory that is constant or zero would make the comparison
        # vacuous, and would look exactly like "no accumulation".
        print(f"  memory readout: norm/step {mem.norm(dim=-1).mean():.3f}, "
              f"across-episode std {mem.std(dim=1).mean():.4f}, "
              f"across-time std {mem.std(dim=0).mean():.4f}")
        print(f"  POSITIVE CONTROL, trial index from memory alone: "
              f"held-out {trial_index_control(mem, n_trials, trial_len):.3f} "
              f"(chance {1.0 / n_trials:.3f})")
        for use_mem in (False, True):
            held, insample = fit_probe(
                agent, enc, mem, targets, stripped, use_mem,
                args.probe_steps, n_trials, trial_len,
            )
            results[(label, use_mem)] = held
            results[(label, use_mem, "in")] = insample
            tag = 'obs+memory' if use_mem else 'obs only  '
            print(f"  probe {tag}: held-out {held[0]:.3f}   in-sample {insample[0]:.3f}")

    print("\n" + "=" * 74)
    print("per-trial potion accuracy (chance .167, memoryless ceiling .1675)")
    print("=" * 74)
    print(f"{'condition':22s} " + "  ".join(f"{('t%d' % i):>5s}" for i in range(n_trials)))
    for key, (acc, curve) in results.items():
        label, use_mem = key[0], key[1]
        scope = "in-sample" if len(key) == 3 else "held-out"
        name = f"{label} {'obs+mem' if use_mem else 'obs only'} {scope}"
        print(f"{name:34s} " + "  ".join(f"{c:5.3f}" for c in curve))
    print()
    print(f"{'condition':34s} " + "  ".join(f"{('t%d' % i):>5s}" for i in range(n_trials)))
    for label in ("trained", "random"):
        base = results[(label, False)][1]
        full = results[(label, True)][1]
        gap = full - base
        print(f"{label:8s} memory contribution (obs+mem - obs): "
              f"t0 {gap[0]:+.3f} -> t{n_trials-1} {gap[-1]:+.3f}   "
              f"slope {np.polyfit(np.arange(n_trials), gap, 1)[0]:+.4f}/trial")
    print("\nrising on random + flat on trained -> exploration collapse")
    print("flat on both                       -> dilution")


if __name__ == "__main__":
    main()
