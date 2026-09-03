from configs.rl.update_fns import update_fn
from ml_collections import ConfigDict

    
def dqn_update_fn(
    config: ConfigDict, max_episode_steps: int, max_training_steps: int
) -> ConfigDict:
    config = update_fn(config)
    # set eps = 1/T, so that the asymptotic prob to
    # sample fully exploited trajectory during exploration is
    # (1-1/T)^T = 1/e
    config.init_eps = 1.0
    config.end_eps = 1.0 / max_episode_steps
    config.schedule_steps = config.schedule_end * max_training_steps

    return config


def get_config():
    config = ConfigDict()
    config.update_fn = dqn_update_fn

    config.algo = "dqn"

    config.critic_lr = 1e-4

    config.config_critic = ConfigDict()
    config.config_critic.hidden_dims = (256, 256)

    config.discount = 0.99
    config.tau = 0.003
    config.schedule_end = 0.1  # at least good for TMaze-like envs

    config.replay_buffer_num_episodes = 1e4

    # Use PopArt value normalization (https://arxiv.org/abs/1809.04474), following AMAGO (https://arxiv.org/abs/2411.11188).
    config.use_popart = False
    config.popart_beta = 5e-4
    config.popart_init_nu = 100.0

    # Symbolic Alchemy only: mask actions for absent stones or potions.
    config.mask_alchemy_invalid_actions = False

    # Symbolic Alchemy only, and only on top of the above: also forbid NO_OP
    # unless no other action is legal. The env accepts NO_OP unconditionally,
    # so this is a policy-side restriction rather than a legality fix. Under a
    # uniform-over-valid policy only 23.75% of steps have NO_OP as the only
    # option, while trained runs idle 37-54% of the time -- and the runs that
    # collapse are the ones that idle most. Applied to action selection and to
    # the target-Q bootstrap alike.
    config.mask_alchemy_no_op = False

    # Symbolic Alchemy only: replace the flat 40-way critic with a dueling +
    # factored head over NO_OP + stone(3) x target(13). Strictly no less
    # expressive than the flat head; adds per-stone and per-target advantage
    # terms so a fact about one stone is shared across its 13 targets.
    # See policies/models/action_heads.py.
    config.factored_action_head = False

    # Symbolic Alchemy only, and only with config_env.aux_canon_target=True:
    # weight on the auxiliary supervised loss that asks the SHARED joint
    # embedding to predict the canonical-frame (latent) stone coordinates and
    # potion types. 0.0 = feature off; the aux head is then not built at all,
    # so the run is bit-identical to the pre-feature code path.
    # See configs/envs/alchemy.py:aux_canon_target for why this adds no
    # information in the oracle configuration.
    config.aux_canon_weight = 0.0

    # Which half of that target to supervise: "both" | "stone" | "potion".
    # The two halves are not the same problem. Measured by
    # scripts/probe_frame_map.py, a memoryless MLP given one observation and no
    # chemistry already reaches 0.756 on stone coordinates (chance 0.5) but
    # only 0.1675 on potion types (chance 0.1667) -- so the stone half is
    # largely free from a single frame while the potion half carries
    # essentially all of the memory-dependent signal. "potion" drops the stone
    # outputs from the head entirely and spends the whole aux gradient there.
    config.aux_canon_parts = "both"

    # WHERE that aux head attaches: "joint" | "memory".
    #   "joint"  -- the critic's own input, conditioner(encoded_obs, h_t). The
    #               aux gradient reaches the memory only through the critic's
    #               trunk, so one set of parameters must serve both the
    #               chemistry target and the value function.
    #   "memory" -- the memory readout h_t alone, excluding the context tail
    #               (which is the oracle's answer key and would let the head
    #               succeed without using memory at all).
    # These are different experiments. With "joint", MATE demonstrably LEARNS
    # the potion permutation (0.567 accuracy against a 0.1675 memoryless
    # ceiling) but return falls 150.4 -> 122.6 at weight 1 -- a representation
    # trade-off in the shared trunk. "memory" tests whether that trade-off is
    # caused by the sharing. Requires a seq model with memory; markov/oracle
    # raises rather than silently training on a zero-width readout.
    config.aux_canon_site = "joint"

    return config
