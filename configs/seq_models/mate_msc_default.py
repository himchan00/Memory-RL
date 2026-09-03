"""MATE + MSC (sub-multiset contrastive) config.

Extends mate_default with the MSC auxiliary loss (policies/seq_models/msc_aux.py;
see mate_context_extraction.md / msc_methodology.md). Defaults encode the
validated recipe: trainable encoder (n_layer=1) + temporal view (prefix vs
suffix, saturation-resistant) + feature gains on the policy path.

Plain MATE runs should use mate_default.py — it carries no msc_* keys.
(Mate's msc_* kwargs default to msc_enable=False, so both configs work.)
"""
from configs.seq_models.mate_default import get_config as mate_get_config


def get_config():
    config = mate_get_config()

    # Validated MSC recipe uses one post-projection encoder layer.
    config.seq_model.n_layer = 1

    # MSC — sub-multiset contrastive. λ·InfoNCE is added to the RL loss through
    # the seq model's `_aux_loss` channel (single backward, shared grad clipping —
    # keep λ ≤ 0.1 or the clipped RL gradient gets crushed).
    config.seq_model.msc_enable          = True
    config.seq_model.msc_lambda          = 0.05  # aux weight λ (total = L_RL + λ·L_MSC)
    config.seq_model.msc_beta            = 0.7   # Bernoulli keep-prob β of the positive view (subset/split only; 0.9 for oracle-type envs like T-Maze)
    config.seq_model.msc_learn_gains     = True  # False: log_gains is a zero buffer (s ≡ 1, policy path untouched) — used by the ladder runs
    config.seq_model.msc_tau             = 0.1   # InfoNCE temperature
    config.seq_model.msc_n_anchors       = 4     # anchor timesteps per episode
    config.seq_model.msc_proj_dim        = 128   # projection head output dim
    config.seq_model.msc_min_anchor_frac = 0.1   # anchors sampled in [frac·L, L-1]
    config.seq_model.msc_detach_z        = True  # variant (i): aux grads reach only gains + head (T-independent backward)
    # View family (see msc_aux.py): subset (full anchor vs beta-subset) |
    # split (disjoint b/(1-b) halves; beta = split ratio, use 0.5) |
    # temporal (prefix [0,t] vs suffix (t,L]; beta unused) |
    # prefix (two prefix means m_t vs m_s; beta unused) |
    # transition (two single transition embeddings z_i vs z_j; beta unused,
    #             requires msc_detach_z=False).
    # All but subset use symmetric InfoNCE. temporal resists saturation
    # (Bayes-error floor) — validated best on ant-dir; transition is floored by
    # the single-transition Bayes error (highest floor of all views).
    config.seq_model.msc_view            = "temporal"
    config.seq_model.msc_focal_gamma     = 0.0   # >0: focal weighting (1-p)^gamma — easy pairs self-erase, gradient stays on hard pairs
    config.seq_model.msc_anchor_power    = 1.0   # >1: oversample EARLY anchor timesteps (t = lo + u^power·(hi-lo))
    config.seq_model.msc_pair_gap        = 0     # transition view only: enforce |i-j| >= gap, damping the state-autocorrelation shortcut (0 = independent draws)

    return config
