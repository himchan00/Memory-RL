from ml_collections import ConfigDict
from typing import Tuple
from gymnasium.envs.registration import register

from configs.envs.common import base_config

# Short alias -> dm_alchemy symbolic level_name. dm_alchemy parses the level_name
# by substring (perceptual_mapping_randomized / rotation / random_bottleneck |
# bottleneck1..3), so there is no fixed registry — these are the useful combos.
LEVELS = {
    "rotation_random_bottleneck":
        "perceptual_mapping_randomized_with_rotation_and_random_bottleneck",
    "random_bottleneck":
        "perceptual_mapping_randomized_with_random_bottleneck",
    "rotation_no_bottleneck":
        "perceptual_mapping_randomized_with_rotation",
    "no_bottleneck":
        "perceptual_mapping_randomized",
    "rotation_bottleneck1":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck1",
    "rotation_bottleneck2":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck2",
    "rotation_bottleneck3":
        "perceptual_mapping_randomized_with_rotation_and_bottleneck3",
    "all_fixed":
        "all_fixed_with_no_bottleneck",
    "all_fixed_rotation":
        "all_fixed_with_rotation",
}


def create_fn(config: ConfigDict) -> Tuple[ConfigDict, str]:
    env_name = config.env_name
    assert env_name in LEVELS, (
        f"Invalid alchemy env_name: {env_name}. Choose from {list(LEVELS.keys())}.")
    register(
        env_name,
        entry_point="envs.alchemy:SymbolicAlchemyEnv",
        max_episode_steps=config.num_trials * config.max_steps_per_trial,
        kwargs=dict(
            level_name=LEVELS[env_name],
            num_trials=config.num_trials,
            max_steps_per_trial=config.max_steps_per_trial,
            observe_used=config.observe_used,
            add_trial_flag=config.add_trial_flag,
            canonicalize_oracle=config.canonicalize_oracle,
            structured_potions=config.structured_potions,
            structured_stones=config.structured_stones,
            add_trial_phase=config.add_trial_phase,
            aux_canon_target=config.aux_canon_target,
            context_graph_only=config.context_graph_only,
            canon_potion_acc=config.canon_potion_acc,
        ),
    )

    del config.create_fn
    return config, env_name


# --- Alchemy's own RL/seq defaults ------------------------------------------
# These live here, not in configs/rl or configs/seq_models, because those are
# shared by T-Maze, MuJoCo, Metaworld and CARL: raising `tau` or turning on
# positional encoding globally would silently re-tune every other environment.
#
# Every value below was measured on this task. tau/use_pe/max_norm are the three
# knobs that separated the 297.8 oracle from the 274.3 port -- restoring all
# three recovered +13.2 points, with use_pe and tau contributing about equally
# (see docs/alchemy_trials.md). PopArt is not optional here: without it the
# Alchemy return scale drives q from 83 to 4305. lr 3e-5 is the value every
# successful run has used; the 1e-4 shared default diverges.
#
# An explicit --config_rl.* / --config_seq.* flag always wins: the override is
# applied only to keys the command line did not mention, so a sweep over tau
# still sweeps tau.
ALCHEMY_RL_DEFAULTS = {
    "tau": 0.003,
    "critic_lr": 3e-5,
    "use_popart": True,
}
ALCHEMY_SEQ_DEFAULTS = {
    "use_pe": True,
    "max_norm": 0.2,
}


def apply_defaults_fn(config_rl, config_seq, explicit):
    """Fill in Alchemy's defaults for anything the command line left alone.

    `explicit` is the set of dotted flag names seen on argv (e.g.
    "config_rl.tau"), so this can never clobber a deliberate override.
    """
    for key, value in ALCHEMY_RL_DEFAULTS.items():
        if f"config_rl.{key}" not in explicit:
            config_rl[key] = value
    for key, value in ALCHEMY_SEQ_DEFAULTS.items():
        if f"config_seq.{key}" not in explicit:
            config_seq[key] = value
    return config_rl, config_seq


def get_config():
    config = base_config()
    config.create_fn = create_fn
    config.apply_defaults_fn = apply_defaults_fn

    config.env_type = "alchemy"
    config.horizon = "finite"  # finite or infinite

    # Symbolic Alchemy meta-episode: `num_trials` trials share one hidden
    # chemistry. Run with --k 1 (the multi-trial structure is native; do NOT use
    # KEpisodeWrapper). The learner reads num_trials to log per-attempt
    # adaptation curves (return_attempt_0..N).
    config.env_name = "rotation_random_bottleneck"  # see LEVELS for choices
    config.num_trials = 10
    config.max_steps_per_trial = 20
    config.observe_used = True
    config.add_trial_flag = True

    # PRIVILEGED, oracle diagnostic only. Rewrites stone coordinates and potion
    # types from the perceptual frame into the latent frame, so the network no
    # longer has to invert the rotation/permutation itself. Use ONLY with
    # --config_seq.seq_model.is_oracle=True; enabling it for a memory model
    # would hand that model the hidden chemistry. See docs/alchemy_status.md P0.
    config.canonicalize_oracle = False

    # NOT privileged -- pure re-encoding, so it is fair to use with any agent.
    # Replaces the ordinal potion `type_value` scalar with axis one-hot(3) +
    # direction(1). Widens symbolic_obs 39 -> 75.
    config.structured_potions = False

    # NOT privileged -- the stone-block twin of structured_potions. An empty
    # stone slot writes the 2.0 absent-sentinel into the three coordinate
    # channels (otherwise -1/0/+1) and the reward channel (otherwise in
    # [-1, 1]); this zeroes those fields and leaves absence signalled solely by
    # the used flag, matching the convention structured_potions already uses.
    # Observation width is UNCHANGED (39 stays 39, or 75 with structured
    # potions), so no downstream rewiring is needed.
    config.structured_stones = False

    # NOT privileged -- appends (steps_left_in_trial, trials_left), both
    # normalized to [0, 1]. add_trial_flag only spikes on the FIRST step of a
    # trial and use_pe only gives the absolute step index, so nothing in the
    # observation directly answers "how long until this trial resets and I lose
    # my un-cashed stones". Widens the observation by 2.
    config.add_trial_phase = False

    # Appends a 21-dim SUPERVISION TARGET (not an input) to the observation:
    # the 3 stones' latent coordinate triples (9) and the 12 potions' latent
    # type indices (12), with AUX_CANON_ABSENT in unoccupied slots. The agent
    # excises this block before RNN_head / the critic / the action mask ever
    # see the observation, and trains an auxiliary head on the shared joint
    # embedding against it (weight: config_rl.aux_canon_weight).
    #
    # NO EXTRA INFORMATION, TRAINING SIGNAL ONLY -- for the ORACLE config.
    # scripts/probe_frame_map.py shows the perceived -> latent map is a
    # deterministic function of (perceived obs, chem_gt[12:28]), learnable to
    # 100% test accuracy by the very critic MLP we use, within one epoch. The
    # oracle already has both inputs; the scalar TD signal simply never drives
    # it to compute the map, which is why the perceived-frame oracle plateaus
    # at ~156 while canonicalize_oracle=True reaches 225+. This makes that
    # function an explicit dense target instead of a hoped-for by-product.
    #
    # CAVEAT: for a MEMORY model (no chem_gt in the observation) the same
    # target IS privileged. Different question -- do not conflate them.
    # Mutually exclusive with canonicalize_oracle (the target would be the
    # identity); the env raises if both are set.
    config.aux_canon_target = False

    # PRIVILEGED, and only valid with canonicalize_oracle=True: keeps chem_gt
    # dims 0-11 (the graph) and drops 12-27 (the frame maps), which are
    # redundant once the observation is already in the latent frame.
    config.context_graph_only = False

    # DIAGNOSTIC, only valid with canonicalize_oracle=True. Probability that
    # each latent potion type is reported CORRECTLY by the canonicalization;
    # 1.0 is the exact identity and leaves the default path untouched.
    #
    # This exists to ask a question about MATE using the oracle. An
    # aux-supervised MATE reaches train/aux_canon_potion_acc = 0.553
    # (mo_site_w1, 24k episodes) but its return does not rise: 146.3 against
    # 150.4 with the aux loss off. Two explanations, indistinguishable from
    # MATE's own numbers -- either 55% is simply too inaccurate to plan with
    # (planning chains facts, so accuracy multiplies), or 55% would be enough
    # and the policy is failing to use it. Setting this to 0.553 on the oracle
    # (232.3 at 1.0) separates them: a collapse toward the 145.2 floor means
    # the accuracy is the binding constraint, and survival means it is not.
    #
    # The map is drawn once per episode, so the agent faces a consistent wrong
    # belief rather than averageable noise. Only the potion half is degraded,
    # which makes this an upper bound for a memory model at the same accuracy.
    config.canon_potion_acc = 1.0

    return config
