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
    config.seq_model.n_layer = 1                # 2 for metaworld, 1 for others
    config.seq_model.hidden_size = 256
    config.seq_model.truncated_sampling = "window"  # subset | window

    config.seq_model.learn_init_emb = True            # initial-memory prior: m_t=(w * init_emb + sum E)/(w + t)
    config.seq_model.use_ema_init_emb = False         # track init_emb as an EMA of valid training transition embeddings
    config.seq_model.ema_init_emb_beta = 5e-4
    config.seq_model.use_rollout_z_cache = False       # reconstruct omitted prefixes from cached rollout embeddings

    # Per-transition gate: m_t = (init + sum_i w_i z_i) / (w_0 + sum_i w_i).
    # Weights numerator and denominator alike, so the memory is still a mean --
    # bounded, order-invariant -- but no longer a UNIFORM one.
    # gate_sparsity_weight pushes mean(w) toward gate_sparsity_target, stating
    # the prior that only a small fraction of transitions carry the context
    # (12 of 200 in Alchemy) instead of hoping the RL loss discovers it.
    config.seq_model.use_gate = False
    config.seq_model.gate_sparsity_weight = 0.0
    config.seq_model.gate_sparsity_target = 0.06

    return config
