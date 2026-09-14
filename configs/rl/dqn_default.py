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

    # Contrastive version of the aux_canon label (policies/models/aux_cpc.py):
    # InfoNCE between the memory read-out and the label instead of regression.
    # Shares aux_canon_site / aux_canon_parts, so only the objective differs.
    config.aux_cpc_weight = 0.0        # 0 = off
    config.aux_cpc_tau = 0.1           # initial temperature; learned
    config.aux_cpc_proj_dim = 128

    # Which half of that target to supervise: "both" | "stone" | "potion".
    # The two halves are not the same problem. Measured by
    # scripts/probe_frame_map.py, a memoryless MLP given one observation and no
    # chemistry already reaches 0.756 on stone coordinates (chance 0.5) but
    # only 0.1675 on potion types (chance 0.1667) -- so the stone half is
    # largely free from a single frame while the potion half carries
    # essentially all of the memory-dependent signal. "potion" drops the stone
    # outputs from the head entirely and spends the whole aux gradient there.
    config.aux_canon_parts = "both"

    # WHERE that aux head attaches: "joint" | "memory" | "memory_obs".
    #   "joint"      -- the critic's own input, conditioner(encoded_obs, h_t).
    #                   The aux gradient reaches the memory only through the
    #                   critic's trunk, so one set of parameters must serve
    #                   both the chemistry target and the value function.
    #   "memory"     -- the memory readout h_t alone, excluding the context
    #                   tail (which is the oracle's answer key and would let
    #                   the head succeed without using memory at all).
    #   "memory_obs" -- cat(encoded_obs.detach(), h_t). The head can SEE the
    #                   current frame but no gradient flows into it, so only
    #                   the memory is shaped.
    # These are different experiments. With "joint", MATE demonstrably LEARNS
    # the potion permutation (0.567 accuracy against a 0.1675 memoryless
    # ceiling) but return falls 150.4 -> 122.6 at weight 1 -- a representation
    # trade-off in the shared trunk.
    #
    # "memory" was meant to test whether that trade-off is caused by the
    # sharing, and is MIS-SPECIFIED for this target: the label is the latent
    # identity of whatever occupies each slot RIGHT NOW, which needs the
    # current frame (what is in the slot) as well as the memory (the
    # perceived->latent map). Measured accuracy: obs only 0.1675 (chance
    # 0.1667), memory only 0.261, obs+memory 0.567. So h_t alone cannot
    # express the target; "memory" plateaued at 0.261 and recovered return
    # (140.9 vs 122.6) only by neutralising the aux loss. "memory_obs" is the
    # corrected form and is what should be used to ask the sharing question.
    #   "probe"      -- cat(encoded_obs.detach(), h_t.detach()). MEASUREMENT
    #                   ONLY: the head trains, the agent does not. Use it to
    #                   read `aux_canon_potion_acc` (chance 0.1667, memoryless
    #                   ceiling 0.1675) out of a run whose chemistry knowledge
    #                   is supposed to come from somewhere else -- e.g. an
    #                   aux_count_weight run -- without the measurement
    #                   changing what it measures. The labels are privileged
    #                   for a memory model, which is exactly why no gradient
    #                   may reach it.
    # Both memory sites require a seq model with memory; markov/oracle raises
    # rather than silently training on a zero-width readout.
    config.aux_canon_site = "joint"

    # Symbolic Alchemy only: "Predict: Features" auxiliary counting loss, the
    # one intervention that worked in the Alchemy paper (arXiv:2102.02926
    # §4.3). Two count vectors -- stones per perceived category (27) and
    # potions per perceived type (6) -- regressed off the conditioner's
    # OBSERVATION branch. In that paper these two tasks, and NOT the
    # ground-truth-chemistry task, took symbolic Alchemy "close to the ideal
    # observer benchmark": "the only case in the present study where agents
    # showed respectable meta-learning performance ... without privileged
    # information at test".
    #
    # NOT PRIVILEGED, and not an env feature: the targets are a deterministic
    # function of the agent's own observation, computed on the training side by
    # envs.alchemy.count_targets_from_observation. Nothing is appended to the
    # observation and nothing can leak, so this is fair for MATE/GPT/LSTM as
    # well as the oracle. 0.0 = feature off; the head is not built at all, so
    # the run is bit-identical to the pre-feature code path.
    #
    # Why it should help HERE specifically: every measured gain on this task so
    # far came from making the observation easier to READ, not from a stronger
    # agent (canonicalize_oracle +35, structured_potions +32, while 4x capacity
    # gave +3.4 and a wider net was negative). Counting per category cannot be
    # done without reading every slot the same way and summing -- the
    # permutation-invariant structure a flat MLP over concatenated slots never
    # acquires. It is `structured_potions` asked for through the loss instead
    # of installed in the input layout.
    config.aux_count_weight = 0.0

    # Which count vector to supervise: "both" | "stone" | "potion".
    config.aux_count_parts = "both"

    # Where the counting head reads from:
    #   "obs"        -- conditioner's observation branch (the original site)
    #   "memory"     -- the memory readout h_t alone
    #   "memory_obs" -- cat(encoded_obs.detach(), h_t); gradient into memory only
    #   "probe"      -- memory_obs with the memory detached: measurement only
    #
    # "obs" is NOT what the paper did. In arXiv:2102.02926 §4.1 symbolic
    # observations "were passed directly to the transformer core" and the
    # auxiliary heads hung off that core, i.e. off the memory, with no
    # observation shortcut in between. The measured failure of the "obs" site
    # here (ledger §6.4: neutral at weight 0.1, -45.2 at weight 1.0, with the
    # counting task itself solved to 0.006 slots of error) is consistent with
    # that difference: counting from the CURRENT observation is a near-linear
    # read that demands no representation work, while counting from the
    # MEMORY requires accumulating slot occupancy across the episode.
    config.aux_count_site = "obs"

    # Symbolic Alchemy only. Concatenate the auxiliary head's DECODED potion
    # posterior (12 slots x 6 types, softmax, detached) onto the critic input.
    #
    # Why: with aux_canon_site="memory_obs" the memory reaches 0.43 held-out on
    # the potion map, yet the policy's useless-potion rate (49.2%) is
    # indistinguishable from uniform-over-legal (49.8%) -- the knowledge is in
    # the memory and never reaches the behaviour. The probe that extracts it is
    # a 256x256 MLP doing nothing else, while the critic must decode AND value
    # 40 actions. This hands the critic the decode already done.
    #
    # Detached, so the decoder is shaped by the auxiliary loss alone. Requires
    # aux_canon_site="memory_obs" and a potion-bearing aux_canon_parts.
    # Diagnostic, not a fair method: the decoder is trained on privileged
    # labels, exactly like the memory_obs runs it is compared against.
    config.aux_canon_feed_critic = False

    return config
