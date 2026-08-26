"""MATE with subset-split bilinear MSC."""

from configs.seq_models.mate_default import get_config as mate_get_config


def get_config():
    config = mate_get_config()

    config.seq_model.msc_enable = True
    config.seq_model.msc_objective = "v2"
    config.seq_model.msc_lambda = 0.05
    config.seq_model.msc_tau = 0.03
    config.seq_model.msc_k_min = 8
    config.seq_model.msc_k_max = 64
    config.seq_model.msc_detach_z = False

    return config
