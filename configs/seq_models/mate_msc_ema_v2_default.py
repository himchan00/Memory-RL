"""MATE with alternating subset-split bilinear MSC."""

from configs.seq_models.mate_msc_v2_default import get_config as mate_msc_get_config


def get_config():
    config = mate_msc_get_config()

    config.seq_model.msc_update_mode = "alternating_ema"
    config.seq_model.msc_updates_per_rl = 1
    config.seq_model.msc_lr = 1e-4

    return config
