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
    config.max_norm = 0.2

    # Compile the agent's CUDA training-loss graph; rollout remains eager.
    config.compile = True

    # fed into RNN_head
    config.obs_shortcut = True
    config.full_transition = True
    config.normalize_inputs = True   # external InputNorm on encoded obs + transition tuple
    config.noise_ratio = 0.0         # Gaussian noise in normalized feature units; requires normalize_inputs=True
    # Absolute-position sinusoidal PE added to the memory readout h_t (RNN_head-level,
    # seq-model-agnostic). Gives the value head an explicit time signal for the finite-
    # horizon RL^2 value "sawtooth". Requires seq_model.max_seq_length.
    config.use_pe = False
    # Skip only the synthetic transition into a soft-reset observation when updating
    # sequence memory (used for k-shot / within-episode resets).
    config.skip_reset_transition = False
    # When skip_reset_transition=True, also drop that same synthetic transition from
    # DQN/SAC RL losses. Disable for memory-only skipping while keeping the reward row.
    config.mask_rl_loss_on_reset_transition = True

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

    # Oracle diagnostic. By default an oracle run gets `perceived_obs ++ chem_gt`
    # concatenated into one flat vector, and the MLP has to work out on its own
    # that chem_gt's frame-map dims are instructions for reinterpreting the
    # perceived dims. Measured 2026-09-01: it cannot -- the concat oracle stops
    # at 156 while the same oracle with the inversion done for it
    # (canonicalize_oracle) reaches 233. This routes chem_gt to the CONDITIONER
    # instead, so with conditioning="hypernet" the context generates the weights
    # that transform the observation. Requires markov + is_oracle + obs_shortcut.
    config.context_as_condition = False

    # Symbolic Alchemy only: replace the flat read of the observation with a
    # shared per-slot MLP over the 3 stone and 12 potion slots, so "how to read
    # a slot" is learned once instead of once per input offset. Per-slot
    # embeddings are kept in slot order (the action head is positional) with a
    # mean-pooled summary appended. See policies/models/slot_encoder.py.
    # Mutually exclusive with use_image_encoder.
    config.alchemy_slot_encoder = False
    config.alchemy_slot_dim = 32
    config.alchemy_slot_hidden_dim = 64

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

    return config
