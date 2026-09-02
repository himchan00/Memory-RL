"""Opt-in alternating MATE + MSC EMA config."""

from configs.seq_models.mate_msc_default import get_config as mate_msc_get_config


def get_config():
    config = mate_msc_get_config()

    # Optimize raw InfoNCE with its own optimizer, including the transition encoder.
    config.seq_model.msc_update_mode = "alternating_ema"
    config.seq_model.msc_detach_z = False
    # Representation-step InfoNCE flavour:
    #   legacy      -- episode-level views from msc_view (MSCAux)
    #   v2          -- disjoint-subset bilinear CPC (MSCV2Aux; usually via the v2 config)
    #   conditional -- query-conditioned InfoNCE (MSCCondAux): anchor = (LOO memory,
    #                  query x_b), candidates = all T*B responses x_t in the batch.
    #                  Blocks the observation-shortcut; lower-bounds I(X_t; Z | X_b).
    config.seq_model.msc_objective = "legacy"
    # Draw one independent replay update for MSC per RL update.
    config.seq_model.msc_updates_per_rl = 1
    # Use this learning rate for MSC; its encoder EMA supplies the policy path.
    config.seq_model.msc_lr = 1e-4

    # Representation-step objective:
    #   L_rep = msc_nce_weight * InfoNCE + msc_recon_beta * LOO-reconstruction
    # (1.0, 0.0) is the pure-contrastive default and reproduces prior runs exactly.
    # (0.0, 1.0) is the pure predictive-sufficiency objective: a decoder predicts
    # the response (r, d_obs) of every transition from the memory that EXCLUDES it.
    # The reconstruction term has no anchor count and no temperature to tune.
    config.seq_model.msc_nce_weight  = 1.0
    config.seq_model.msc_recon_beta  = 0.0
    config.seq_model.msc_recon_hidden = 256
    # all | reward | dynamics -- which response channels the decoder predicts.
    # Reward-only suits reward-varying families (ant-dir, cheetah-vel) where the
    # dynamics channel provably carries zero task information; dynamics-only suits
    # parameter-varying families (walker-param, hopper-param).
    config.seq_model.msc_recon_target = "all"
    # mlp    : learned throwaway decoder (~122k params, same scale as the InfoNCE head)
    # linear : NO learnable module -- ridge regression solved in closed form per
    #          batch, fit on half the episodes and scored on the other half.
    #          Enforces MATE's premise that the response is LINEARLY readable
    #          from the mean embedding.
    config.seq_model.msc_recon_decoder = "mlp"
    config.seq_model.msc_recon_ridge = 1e-3

    return config
