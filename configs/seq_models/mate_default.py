from ml_collections import ConfigDict
from configs.seq_models.common import base_config
from configs.seq_models.update_fns import update_fn


def mate_update_fn(config: ConfigDict, max_episode_steps: int) -> ConfigDict:
    config = update_fn(config, max_episode_steps)

    config.seq_model.max_seq_length = (
        max_episode_steps + 1
    )  # NOTE: transition data starts from t=1

    return config


def get_config():
    config = base_config()
    config.update_fn = mate_update_fn
    
    # MATE-specific defaults
    config.obs_shortcut = True

    # seq_model specific
    config.seq_model.name = "mate"
    config.seq_model.use_gate = False           # True uses gate, False uses simple sum
    config.seq_model.n_layer = 1                # 2 for metaworld, 1 for others
    config.seq_model.hidden_size = 256          # 256 default; overridden to 128 for tmaze envs via CLI
    config.seq_model.gate_noise_std = 0.0       # std of Gaussian noise on gate logits (0 = disabled)
    config.seq_model.init_emb_zero = False      # ablation: if True, init_emb fixed to zeros (non-trainable buffer)
    config.seq_model.rollout_dropout = 0.0
    config.seq_model.transition_dropout = 0.0

    config.seq_model.use_rff = False            # if True, last embedding layer is RFFEmbedding (kernel-mean MATE)
    config.seq_model.kernel = "gaussian"        # gaussian | laplace | matern | train (only when use_rff=True)
    config.seq_model.learnable_lambda = False   # if True, ARD per-dim scaling in RFFEmbedding (length-scale ∝ 1/lambda)

    return config
