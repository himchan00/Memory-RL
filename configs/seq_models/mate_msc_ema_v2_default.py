"""MATE with alternating subset-split bilinear MSC."""

from configs.seq_models.mate_msc_v2_default import get_config as mate_msc_get_config


def get_config():
    config = mate_msc_get_config()

    config.seq_model.msc_update_mode = "alternating_ema"
    config.seq_model.msc_updates_per_rl = 1
    config.seq_model.msc_lr = 1e-4
    # v3 (scale-covering CPC) only: subtract the detached query-twin logits
    # from the main logits (shortcut correction). Keep False unless the
    # logged msc_accq is well above chance in the target environment.
    config.seq_model.msc_v3_twin_correction = False

    # Representation-step objective (shared with the legacy EMA config):
    #   L_rep = msc_nce_weight * InfoNCE + msc_recon_beta * LOO-reconstruction
    # Here the InfoNCE term is the v2 subset-split bilinear CPC.
    # (1.0, 0.0) is the pure-contrastive default; (0.0, 1.0) drops CPC entirely
    # and trains the encoder by predictive sufficiency alone.
    config.seq_model.msc_nce_weight  = 1.0
    config.seq_model.msc_recon_beta  = 0.0
    config.seq_model.msc_recon_hidden = 256
    # all | reward | dynamics -- which response channels the decoder predicts.
    config.seq_model.msc_recon_target = "all"
    # mlp    : learned throwaway decoder
    # linear : NO learnable module -- closed-form ridge, fit on half the episodes
    #          and scored on the other half.
    config.seq_model.msc_recon_decoder = "mlp"
    config.seq_model.msc_recon_ridge = 1e-3

    return config
