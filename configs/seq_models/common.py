"""Shared defaults for all `configs/seq_models/*.py` configs.

Each specific config calls `base_config()`, then overrides what's specific to
that sequence model — e.g. markov flips `full_transition` to False.

For pixel-based environments, toggle `config.use_image_encoder = True` via CLI
(`--config_seq.use_image_encoder=True`). The `image_encoder.*` defaults are
always attached so they can be overridden on the command line.

Keep this file in sync with `policies/models/recurrent_head.py` (which is the
sole consumer of these `config_seq.*` keys).
"""
from ml_collections import ConfigDict


def base_config() -> ConfigDict:
    """Common `config_seq.*` defaults shared by every sequence model."""
    config = ConfigDict()

    # gradient clipping (applied by Learner)
    config.clip = True
    config.max_norm = 0.1

    # Compile the agent's CUDA training-loss graph; rollout remains eager.
    config.compile = True

    # fed into RNN_head
    config.obs_shortcut = True
    config.full_transition = True
    config.normalize_inputs = True   # external InputNorm on encoded obs + transition tuple
    config.project_output = False    # project obs + memory readouts to radius sqrt(D)
    config.noise_ratio = 0.0         # Gaussian noise in normalized feature units; requires normalize_inputs=True
    # Absolute-position sinusoidal PE added to the memory readout h_t (RNN_head-level,
    # seq-model-agnostic). Gives the value head an explicit time signal for the finite-
    # horizon RL^2 value "sawtooth". Requires seq_model.max_seq_length.
    config.use_pe = False

    # Dropout policy (amago-style: dropout_emb on input embedding,
    # dropout_ff on feed-forward layers, not applied to actor/critic networks).
    config.dropout_emb = 0.05
    config.dropout_ff = 0.05

    # FiLM / Hypernet conditioning (see policies/models/conditioning.py)
    config.conditioning = "concat"          # "concat" | "film" | "hypernet"
    # Conditioner depth: n_layer modulated blocks (film/hypernet) or n_layer hidden
    # layers added after linear projection layer.
    # Per-mode layout:
    #   concat   → Linear+act(in→h), then n_layer × (Linear → act), then cat(out, c)
    #   film     → Linear+act(in→h), then n_layer × (Linear → act → FiLM(·, c))
    #   hypernet → Linear+act(in→h), then n_layer × (HyperLinear(·, c) → act)
    config.conditioning_n_layer = 1
    # Conditioner hidden/output width. Decoupled from seq_model.hidden_size.
    config.conditioning_hidden_dim = 256

    # Image encoder toggle + defaults (active only when use_image_encoder=True).
    # The standard 96x96 Atari-style conv stack used by every pixel-based env.
    config.use_image_encoder = False
    config.image_encoder = ConfigDict()
    config.image_encoder.image_shape = (3, 96, 96)
    config.image_encoder.embedding_size = 128
    config.image_encoder.channels = (32, 64)
    config.image_encoder.kernel_sizes = (8, 4)
    config.image_encoder.strides = (4, 4)

    # shared seq_model settings; each specific config adds model fields
    config.seq_model = ConfigDict()
    config.seq_model.truncated_sampling = "window"  # window | subset

    return config
