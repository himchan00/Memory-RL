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
    config.max_norm = 1.0

    # torch.compile toggle (applied in RNN_head.__init__)
    config.compile = False

    # fed into RNN_head
    config.obs_shortcut = True
    config.full_transition = True
    config.normalize_inputs = True   # external InputNorm on encoded obs + transition tuple

    # FiLM / Hypernet conditioning (see policies/models/conditioning.py)
    config.conditioning = "concat"          # "concat" | "film" | "hypernet"
    # Conditioner layer dims: (in_dim, *conditioning_hidden_sizes, hidden_size).
    # Same shape across all 3 modes:
    #   concat   → (Linear → act) stack, then cat(out, c)
    #   film     → (Linear → act → FiLM(·, c)) stack
    #   hypernet → (HyperLinear(·, c) → act) stack
    config.conditioning_hidden_sizes = ()

    # Image encoder toggle + defaults (active only when use_image_encoder=True).
    # The standard 96x96 Atari-style conv stack used by every pixel-based env.
    config.use_image_encoder = False
    config.image_encoder = ConfigDict()
    config.image_encoder.image_shape = (3, 96, 96)
    config.image_encoder.embedding_size = 128
    config.image_encoder.channels = (32, 64)
    config.image_encoder.kernel_sizes = (8, 4)
    config.image_encoder.strides = (4, 4)

    # seq_model.* is populated by the specific config
    config.seq_model = ConfigDict()

    # (transition, observation) embedder defaults
    config.embedder = ConfigDict()
    config.embedder.hidden_sizes = ()
    config.embedder.norm = "none"
    config.embedder.output_activation = "leakyrelu"

    return config
