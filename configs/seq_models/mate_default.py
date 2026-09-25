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
    config.seq_model.embedder_type = "mlp"      # "mlp": n_layer x (Linear->LeakyReLU->Dropout) | "gpt_ffn": n_layer x GPT-2 residual FFN block + final LayerNorm
    config.seq_model.hidden_size = 256

    config.seq_model.learn_init_emb = True            # initial-memory prior: m_t=(w * init_emb + sum E)/(w + t)
    config.seq_model.use_ema_init_emb = False         # track init_emb as an EMA of valid training transition embeddings
    config.seq_model.ema_init_emb_beta = 5e-4
    config.seq_model.use_store = False                # STORE: subset training over reused embeddings (--config_seq.seq_model.use_store=True)
    config.seq_model.store_grad_correction = True     # rescale the reused-embedding gradient by (T-1)/(k-1)
    config.seq_model.store_fresh_target = True         # successor memory (target input) also gets the fresh deltas; False (cached z only) lets raw_grad_norm grow 7-8x over a run
    config.seq_model.transition_sampling_method = "epoch"  # STORE re-embedded rows: "epoch" (permutation blocks) | "iid" (fresh random subset)
    config.seq_model.store_independent_loss_rows = True   # sample actor/critic loss rows (iid random) independently of the re-embedded rows
    config.seq_model.store_cache_ema_beta = 1.0       # per-update EMA rate of the cache (row unseen for D updates: weight 1-(1-beta)^D); 1.0 = replace

    return config
