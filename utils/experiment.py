import sys
from collections.abc import Mapping


def explicit_config_flags(argv=None):
    """Dotted config flag names the command line actually mentioned.

    ml_collections accepts both `--config_rl.tau=0.003` and
    `--config_rl.tau 0.003`, so both spellings are recognised. Used to make an
    environment's own defaults yield to a deliberate override.
    """
    argv = sys.argv if argv is None else argv
    names = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        if name.startswith(("config_rl.", "config_seq.")):
            names.add(name)
    return names


def finalize_training_configs(
    config_rl,
    config_seq,
    *,
    max_episode_steps: int,
    train_episodes: int,
    config_env=None,
):
    # An environment may carry its own RL/seq defaults (Alchemy does). They are
    # applied BEFORE the update_fns, so anything derived from them -- the
    # epsilon schedule, max_seq_length -- sees the final values. Keys named on
    # the command line are left untouched.
    if config_env is not None and "apply_defaults_fn" in config_env:
        apply_defaults_fn = config_env.apply_defaults_fn
        del config_env.apply_defaults_fn
        config_rl, config_seq = apply_defaults_fn(
            config_rl, config_seq, explicit_config_flags()
        )

    seq_update_fn = config_seq.update_fn
    rl_update_fn = config_rl.update_fn
    config_seq = seq_update_fn(config_seq, max_episode_steps)
    max_training_steps = int(train_episodes * max_episode_steps)
    config_rl = rl_update_fn(
        config_rl,
        max_episode_steps,
        max_training_steps,
    )
    return config_rl, config_seq


def validate_run_settings(
    config_env,
    *,
    max_seq_len: int,
    max_episode_steps: int,
) -> None:
    n_env = int(config_env["n_env"])
    for field in ("log_interval", "eval_interval", "eval_episodes"):
        value = int(config_env[field])
        if value % n_env != 0:
            raise ValueError(f"{field} must be divisible by n_env")

    if max_seq_len > max_episode_steps:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) must be <= episode length "
            f"({max_episode_steps})"
        )


def validate_resume_config(saved_config, current_config) -> None:
    skip = {
        ("config_rl", "schedule_steps"),
        ("config_rl", "replay_buffer_num_episodes"),
    }
    for section in ("config_env", "config_rl", "config_seq"):
        saved_section = saved_config[section]
        for key, current_value in current_config[section].items():
            if (section, key) in skip or key not in saved_section:
                continue
            _validate_config_value(
                path=f"{section}.{key}",
                saved=saved_section[key],
                current=current_value,
            )


def _validate_config_value(*, path: str, saved, current) -> None:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        for key, current_value in current.items():
            if key in saved:
                _validate_config_value(
                    path=f"{path}.{key}",
                    saved=saved[key],
                    current=current_value,
                )
        return
    if saved != current:
        raise ValueError(f"Config mismatch on resume: {path}")
