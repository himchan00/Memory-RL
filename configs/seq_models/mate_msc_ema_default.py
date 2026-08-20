"""Opt-in alternating MATE + MSC EMA config."""

from configs.seq_models.mate_msc_default import get_config as mate_msc_get_config


def get_config():
    config = mate_msc_get_config()

    # Optimize raw InfoNCE with its own optimizer, including the transition encoder.
    config.seq_model.msc_update_mode = "alternating_ema"
    config.seq_model.msc_detach_z = False
    # Draw one independent replay update for MSC per RL update.
    config.seq_model.msc_updates_per_rl = 1
    # Use this learning rate for MSC; its encoder EMA supplies the policy path.
    config.seq_model.msc_lr = 1e-4

    return config
