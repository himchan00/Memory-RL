import torch
from copy import deepcopy
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from envs.alchemy import (
    AUX_CANON_DIM,
    AUX_CANON_GRAPH_DIM,
    AUX_CANON_NUM_POTION_TYPES,
    AUX_CANON_POTION_DIM,
    AUX_CANON_STONE_DIM,
    AUX_COUNT_POTION_CATEGORIES,
    AUX_COUNT_STONE_CATEGORIES,
    TRIAL_PHASE_DIM,
    count_targets_from_observation,
    get_symbolic_alchemy_layout,
    present_flags_from_observation,
    valid_action_mask_from_observation,
)
from policies.models.aux_cpc import (
    AuxCanonCPC, encode_label, label_width,
)
from policies.models.off_policy_utils import (
    clip_gradients,
    prepare_recurrent_batch,
)
from policies.models.action_heads import FactoredAlchemyQHead
from policies.models.recurrent_head import RNN_head
from policies.models.popart import PopArt
from torchkit.networks import FlattenMlp
import torchkit.pytorch_utils as ptu
from utils.helpers import get_constant_schedule_with_warmup

# 3 stones x 3 coordinate regressions + 12 potions x 6-way type logits.
AUX_CANON_STONE_OUT = AUX_CANON_STONE_DIM                     # 9 coord regressions
AUX_CANON_POTION_OUT = AUX_CANON_POTION_DIM * AUX_CANON_NUM_POTION_TYPES  # 12 x 6 logits
AUX_CANON_GRAPH_OUT = AUX_CANON_GRAPH_DIM               # 12 binary edge logits
AUX_CANON_OUT_DIM = AUX_CANON_STONE_OUT + AUX_CANON_POTION_OUT
# "both" is kept as the legacy name for stone+potion so old commands still mean
# what they meant. "graph" and "potion_graph" are the new options; the second
# is the one the exploration diagnostic argues for.
AUX_CANON_PARTS = ("both", "stone", "potion", "graph", "potion_graph", "all")
AUX_CANON_SITES = ("joint", "memory", "memory_obs", "probe")

# "Predict: Features" auxiliary count head (Alchemy paper §4.3). Two count
# vectors regressed off the conditioner's OBSERVATION branch.
AUX_COUNT_PARTS = ("both", "stone", "potion")
# Where the counting head reads from. The paper (arXiv:2102.02926 §4.1) passed
# symbolic observations "directly to the transformer core" and hung the
# auxiliary heads off that core -- i.e. off the MEMORY, with no observation
# shortcut in between. Our default "obs" site is therefore NOT the paper's:
# it shapes the perceptual branch only, which in symbolic Alchemy is a
# near-linear read of the slot block and so demands almost no work.
#   "obs"        -- the conditioner's observation branch (original behaviour).
#   "memory"     -- the memory readout h_t alone; the paper's site.
#   "memory_obs" -- cat(encoded_obs.detach(), h_t): may READ the current frame
#                   but sends gradient only into the memory.
#   "probe"      -- memory_obs with the memory detached too: measures what the
#                   memory already holds without shaping the agent.
AUX_COUNT_SITES = ("obs", "memory", "memory_obs", "probe")
AUX_COUNT_STONE_OUT = AUX_COUNT_STONE_CATEGORIES
AUX_COUNT_POTION_OUT = AUX_COUNT_POTION_CATEGORIES


class LinearSchedule:
    def __init__(self, init_value: float, end_value: float, transition_steps: int):
        self.init = float(init_value)
        self.end = float(end_value)
        self.n = max(1, int(transition_steps))

    def __call__(self, step: int) -> float:
        t = 0 if step < 0 else self.n if step > self.n else step
        frac = t / self.n
        return (1.0 - frac) * self.init + frac * self.end


class ModelFreeOffPolicy_DQN_RNN(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        config_seq,
        config_rl,
        **kwargs
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = config_rl.discount
        self.tau = config_rl.tau
        self.clip = config_seq.clip
        self.clip_grad_norm = config_seq.max_norm
        self.compile_training_loss = bool(config_seq.get("compile", False))
        self._compiled_compute_loss = None
        self.mask_rl_loss_on_reset_transition = bool(
            config_seq.get("skip_reset_transition", False)
            and config_seq.get("mask_rl_loss_on_reset_transition", True)
        )
        config_env = kwargs.get("config_env")
        is_alchemy = getattr(config_env, "env_type", None) == "alchemy"
        self.mask_alchemy_invalid_actions = bool(
            getattr(config_rl, "mask_alchemy_invalid_actions", False)
            and is_alchemy
        )
        # Additionally forbid NO_OP whenever any real action is available. The
        # env always accepts NO_OP, so this is a policy-side restriction that
        # does not change the environment's legality: measured over 8000 steps
        # under a uniform-over-valid policy, 23.75% of steps have NO_OP as the
        # only legal action, while our runs idle 37-54% of the time -- so a
        # large share of that idling is chosen, not forced, and it is where the
        # collapsing aux runs retreat to. Applied at BOTH action selection and
        # the target-Q bootstrap, so the target never values an action the
        # policy cannot take.
        self.mask_alchemy_no_op = bool(
            getattr(config_rl, "mask_alchemy_no_op", False) and is_alchemy
        )
        if self.mask_alchemy_no_op and not self.mask_alchemy_invalid_actions:
            raise ValueError(
                "config_rl.mask_alchemy_no_op=True requires "
                "config_rl.mask_alchemy_invalid_actions=True (NO_OP is masked "
                "as part of the same mask)"
            )

        ## Auxiliary canonical-frame supervision (Alchemy only).
        # The env appends a 21-dim TARGET block to the observation. It is a
        # label, never an input: `_strip_aux_target` excises it before
        # RNN_head, the critic and the action mask ever see the observation,
        # and `net_obs_dim` (not `obs_dim`) is what the network is built for.
        self.aux_canon_target = bool(
            getattr(config_env, "aux_canon_target", False) and is_alchemy
        )
        self.aux_canon_weight = float(
            getattr(config_rl, "aux_canon_weight", 0.0)
        )
        if self.aux_canon_weight > 0.0 and not self.aux_canon_target:
            raise ValueError(
                "config_rl.aux_canon_weight > 0 requires "
                "config_env.aux_canon_target=True (there is no target to "
                "regress onto otherwise)"
            )
        # weight == 0 means OFF: no aux head is built and no aux term is
        # computed, so the run is bit-identical to the pre-feature code path.
        self.aux_canon_enabled = (
            self.aux_canon_target and self.aux_canon_weight > 0.0
        )
        # Contrastive variant of the SAME label (see policies/models/aux_cpc.py).
        # Independent weight: either objective can run alone or both together.
        self.aux_cpc_weight = float(getattr(config_rl, "aux_cpc_weight", 0.0))
        if self.aux_cpc_weight > 0.0 and not self.aux_canon_target:
            raise ValueError(
                "config_rl.aux_cpc_weight > 0 requires "
                "config_env.aux_canon_target=True"
            )
        self.aux_cpc_enabled = self.aux_cpc_weight > 0.0
        # Which half of the target to supervise. The two halves are NOT the
        # same problem: measured by scripts/probe_frame_map.py, a memoryless
        # MLP on one observation already gets stone coords to 0.756 (chance
        # 0.5) but potion types only to 0.1675 (chance 0.1667). So the stone
        # half is mostly free and the potion half carries essentially all of
        # the memory-dependent signal. "potion" spends the whole aux gradient
        # on the part that needs memory.
        self.aux_canon_parts = str(
            getattr(config_rl, "aux_canon_parts", "both")
        )
        if self.aux_canon_parts not in AUX_CANON_PARTS:
            raise ValueError(
                f"config_rl.aux_canon_parts must be one of {AUX_CANON_PARTS}, "
                f"got {self.aux_canon_parts!r}"
            )
        self.aux_canon_use_stone = self.aux_canon_parts in ("both", "stone", "all")
        self.aux_canon_use_potion = self.aux_canon_parts in (
            "both", "potion", "potion_graph", "all"
        )
        self.aux_canon_use_graph = self.aux_canon_parts in (
            "graph", "potion_graph", "all"
        )
        # WHERE the aux head attaches.
        #   "joint"      -- the critic's own input, conditioner(encoded_obs, h_t).
        #   "memory"     -- the memory readout h_t alone.
        #   "memory_obs" -- cat(encoded_obs.detach(), h_t): the head can SEE
        #                   the current frame but no gradient flows into it,
        #                   so only the memory is shaped.
        #
        # "memory" turned out to be MIS-SPECIFIED and "memory_obs" is the fix.
        # The target is the latent identity of whatever occupies each slot RIGHT
        # NOW, which needs both the current frame (what is in the slot) and the
        # memory (the perceived->latent mapping). Measured potion accuracy:
        #   obs only, no memory   0.1675  (chance 0.1667)
        #   memory only           0.261   <- site="memory", never took off
        #   obs + memory          0.567   <- site="joint"
        # So h_t alone cannot express the target and 0.26 is a structural
        # ceiling, not a training failure. site="memory" therefore recovered
        # return (140.9 vs 122.6) only by neutralising the aux loss, which
        # answers a different question than the one we asked.
        # These are different experiments. On "joint" the aux gradient reaches
        # the memory only THROUGH the critic's trunk, so the same parameters
        # must serve both the chemistry target and the value function; round 6
        # showed that costs the value representation (return 150.4 -> 122.6 at
        # weight 1) even though the potion permutation IS learned (0.567 vs a
        # 0.1675 memoryless ceiling). "memory" pushes the target into the
        # memory without routing it through the critic's input.
        #
        # "probe" is not a training site at all: it is "memory_obs" with the
        # MEMORY detached as well, so the head reads cat(obs, h) but sends
        # gradient into neither. It exists to MEASURE what the memory already
        # contains -- `aux_canon_potion_acc` against a 0.1675 memoryless
        # ceiling -- in runs whose chemistry knowledge is meant to come from
        # somewhere else (e.g. the count auxiliary), without that measurement
        # changing the thing being measured. The labels are privileged for a
        # memory model, which is exactly why the gradient must not reach it.
        ## Feed the DECODED chemistry back into the critic's input.
        #
        # scripts/diagnose_exploration.py measured a memory that decodes the
        # potion map at 0.43 held-out driving a policy whose useless-potion rate
        # (49.2%) is indistinguishable from uniform-over-legal (49.8%). The
        # information is in the memory and does not reach the behaviour. The
        # probe that reads it is a 256x256 MLP doing nothing else; the critic
        # has to do the same decode while also valuing 40 actions.
        #
        # So: hand the critic the decode, already done. The aux head's potion
        # posterior (12 slots x 6 types, softmaxed) is DETACHED and concatenated
        # onto the critic input. If return moves, the decode was the blocker; if
        # it does not, a 0.43-accurate potion map is simply not worth much to
        # this policy and no amount of better memory will help.
        #
        # Requires aux_canon_site="memory_obs" and parts covering the potion
        # half -- the quantity fed forward has to be the one being supervised.
        self.aux_canon_feed_critic = bool(
            getattr(config_rl, "aux_canon_feed_critic", False)
        )
        self.aux_canon_site = str(getattr(config_rl, "aux_canon_site", "joint"))
        if self.aux_canon_site not in AUX_CANON_SITES:
            raise ValueError(
                f"config_rl.aux_canon_site must be one of {AUX_CANON_SITES}, "
                f"got {self.aux_canon_site!r}"
            )

        ## "Predict: Features" auxiliary counting loss (Alchemy only).
        # Targets are computed from the agent's own observation, so unlike
        # aux_canon_target this needs nothing from the env and can never leak.
        self.aux_count_weight = float(
            getattr(config_rl, "aux_count_weight", 0.0)
        )
        self.aux_count_parts = str(
            getattr(config_rl, "aux_count_parts", "both")
        )
        if self.aux_count_parts not in AUX_COUNT_PARTS:
            raise ValueError(
                f"config_rl.aux_count_parts must be one of {AUX_COUNT_PARTS}, "
                f"got {self.aux_count_parts!r}"
            )
        if self.aux_count_weight > 0.0 and not is_alchemy:
            raise ValueError(
                "config_rl.aux_count_weight > 0 is Symbolic Alchemy only "
                "(the targets are counts over its stone/potion slots)"
            )
        # weight == 0 means OFF: no head is built and no term is computed, so
        # the run is bit-identical to the pre-feature code path.
        self.aux_count_enabled = self.aux_count_weight > 0.0
        self.aux_count_use_stone = self.aux_count_parts in ("both", "stone")
        self.aux_count_use_potion = self.aux_count_parts in ("both", "potion")
        self.aux_count_site = str(getattr(config_rl, "aux_count_site", "obs"))
        if self.aux_count_site not in AUX_COUNT_SITES:
            raise ValueError(
                f"config_rl.aux_count_site must be one of {AUX_COUNT_SITES}, "
                f"got {self.aux_count_site!r}"
            )

        if self.aux_canon_feed_critic:
            if not self.aux_canon_enabled:
                raise ValueError(
                    "config_rl.aux_canon_feed_critic requires an enabled aux "
                    "head (aux_canon_target=True and aux_canon_weight > 0)"
                )
            if self.aux_canon_site != "memory_obs":
                raise ValueError(
                    "aux_canon_feed_critic expects aux_canon_site='memory_obs': "
                    "the fed quantity must be decoded from (obs, memory) and "
                    "shaped into the memory, which is what that site does"
                )
        self.aux_canon_feed_dim = (
            AUX_CANON_POTION_OUT if self.aux_canon_feed_critic else 0
        )
        self._aux_start = 0
        self._aux_end = 0
        self.net_obs_dim = self.obs_dim - (
            AUX_CANON_DIM if self.aux_canon_target else 0
        )

        self._alchemy_mask_kwargs = None
        self._alchemy_split_kwargs = None
        if is_alchemy and (
            self.mask_alchemy_invalid_actions
            or self.aux_canon_target
            or self.aux_count_enabled
        ):
            observe_used = bool(getattr(config_env, "observe_used", True))
            add_trial_flag = bool(
                getattr(config_env, "add_trial_flag", False)
            )
            structured_potions = bool(
                getattr(config_env, "structured_potions", False)
            )
            add_trial_phase = bool(
                getattr(config_env, "add_trial_phase", False)
            )
            layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
            symbolic_obs_dim = (
                layout.symbolic_obs_dim
                + int(add_trial_flag)
                + (TRIAL_PHASE_DIM if add_trial_phase else 0)
            )
            # The aux block sits immediately after the trial flag/phase and
            # before the oracle chem_gt tail (see envs/alchemy.py:_split_obs).
            self._aux_start = symbolic_obs_dim
            self._aux_end = symbolic_obs_dim + AUX_CANON_DIM
            # Every downstream split sees the STRIPPED observation, so the
            # context tail is measured against net_obs_dim.
            context_dim = self.net_obs_dim - symbolic_obs_dim
            if context_dim < 0:
                raise ValueError(
                    "Alchemy observation width is smaller than the symbolic "
                    f"layout: obs_dim={self.obs_dim}, "
                    f"net_obs_dim={self.net_obs_dim}, "
                    f"symbolic_obs_dim={symbolic_obs_dim}"
                )
            self._alchemy_split_kwargs = {
                "observe_used": observe_used,
                "add_trial_flag": add_trial_flag,
                "context_dim": context_dim,
                "structured_potions": structured_potions,
                "add_trial_phase": add_trial_phase,
            }
            if self.mask_alchemy_invalid_actions:
                # A SUPERSET, not an alias: `present_flags_from_observation`
                # also consumes `_alchemy_split_kwargs` and does not take
                # `mask_no_op`, so mutating the shared dict would break it.
                self._alchemy_mask_kwargs = {
                    **self._alchemy_split_kwargs,
                    "mask_no_op": self.mask_alchemy_no_op,
                }

        # Trial geometry, for PER-TRIAL auxiliary-accuracy logging. One
        # accuracy averaged over all 200 steps cannot show whether the memory
        # ACCUMULATES inside an episode, which is the only thing a meta-RL
        # memory is there to do: 0.553 could be a flat 0.553 at every trial
        # (the memory never fills) or 0.30 rising to 0.80 (it does). Those two
        # imply opposite next experiments and the averaged number cannot tell
        # them apart.
        self._alchemy_num_trials = int(getattr(config_env, "num_trials", 0) or 0)
        self._alchemy_steps_per_trial = int(
            getattr(config_env, "max_steps_per_trial", 0) or 0
        )

        self.epsilon_schedule = LinearSchedule(
            init_value=config_rl.init_eps,
            end_value=config_rl.end_eps,
            transition_steps=config_rl.schedule_steps,
        )
        self.count = 0

        # Shared RNN encoder — built for the STRIPPED observation width.
        self.head = RNN_head(self.net_obs_dim, action_dim, config_seq, config_env)
        self.alternating_msc = bool(self.head.alternating_msc)
        # NOTE: no target head. Following amago

        # Q-value network
        self.qf = self._build_qf(config_rl, config_env)
        self.qf_target = deepcopy(self.qf)

        # Auxiliary head on the SHARED joint embedding the critic reads.
        # Output layout matches scripts/probe_frame_map.py: 9 stone-coordinate
        # regressions followed by 12 x 6 potion-type logits. A disabled half
        # is dropped from the head entirely rather than masked out of the
        # loss, so no capacity is spent on it and no untrained accuracy is
        # reported for it.
        self.aux_canon_head = None
        if self.aux_canon_enabled:
            if self.aux_canon_site in ("memory", "memory_obs", "probe"):
                # Fail loudly rather than silently training a head on a readout
                # that holds nothing: markov/oracle has no memory, so
                # "supervise the memory" is undefined there. Test the readout
                # width, NOT seq_model.hidden_size -- markov reports
                # hidden_size=256, but that configures its OBSERVATION
                # embedding, not a memory.
                #
                # The guard matters MORE for "memory_obs" than for "memory":
                # there the head also reads the observation, so with no memory
                # it would still fit the target off the frame alone and report
                # a plausible accuracy while supervising nothing.
                if self.head.memory_embed_size <= 0:
                    raise ValueError(
                        f"config_rl.aux_canon_site={self.aux_canon_site!r} "
                        "requires a sequence model with a memory readout, but "
                        f"{self.head.seq_model.name!r} has "
                        f"memory_embed_size={self.head.memory_embed_size} "
                        "(no memory to supervise)"
                    )
                self.head.expose_memory_embeds = True
            if self.aux_canon_site == "memory":
                aux_input_size = self.head.memory_embed_size
            elif self.aux_canon_site in ("memory_obs", "probe"):
                aux_input_size = (
                    self.head.encoded_obs_size + self.head.memory_embed_size
                )
            else:
                aux_input_size = self.head.embedding_size
            self.aux_canon_head = FlattenMlp(
                input_size=aux_input_size,
                output_size=(
                    AUX_CANON_STONE_OUT * int(self.aux_canon_use_stone)
                    + AUX_CANON_POTION_OUT * int(self.aux_canon_use_potion)
                    + AUX_CANON_GRAPH_OUT * int(self.aux_canon_use_graph)
                ),
                hidden_sizes=config_rl.config_critic.hidden_dims,
            )

        # "Predict: Features" counting head, on the conditioner's OBSERVATION
        # branch. NOT `encoded_obs`: for a non-pixel env `_encode_obs` is the
        # identity, so a head there would share no parameters with the network
        # and regularise nothing.
        self.aux_count_head = None
        if self.aux_count_enabled:
            if self.aux_count_site == "obs":
                if self.head.obs_embed_size <= 0:
                    raise ValueError(
                        "config_rl.aux_count_weight > 0 with "
                        "aux_count_site='obs' requires a conditioner to attach "
                        "to; got obs_shortcut=False (no observation branch)"
                    )
                self.head.expose_obs_embeds = True
                count_in_dim = self.head.obs_embed_size
            else:
                if self.head.memory_embed_size <= 0:
                    raise ValueError(
                        f"config_rl.aux_count_site={self.aux_count_site!r} "
                        "needs a sequence model with a memory; got "
                        f"memory_embed_size={self.head.memory_embed_size}"
                    )
                self.head.expose_memory_embeds = True
                count_in_dim = self.head.memory_embed_size
                if self.aux_count_site in ("memory_obs", "probe"):
                    count_in_dim += self.head.encoded_obs_size
            self.aux_count_head = FlattenMlp(
                input_size=count_in_dim,
                output_size=(
                    AUX_COUNT_STONE_OUT * int(self.aux_count_use_stone)
                    + AUX_COUNT_POTION_OUT * int(self.aux_count_use_potion)
                ),
                hidden_sizes=config_rl.config_critic.hidden_dims,
            )

        # PopArt value normalization (no-op when disabled)
        self.popart = PopArt(
            beta=getattr(config_rl, "popart_beta", 5e-4),
            init_nu=getattr(config_rl, "popart_init_nu", 100.0),
            enabled=getattr(config_rl, "use_popart", False),
        )

        # Contrastive head on the same label and the same read-out site.
        self.aux_cpc_head = None
        if self.aux_cpc_enabled:
            if self.aux_canon_site in ("memory", "memory_obs", "probe"):
                if self.head.memory_embed_size <= 0:
                    raise ValueError(
                        f"aux_canon_site={self.aux_canon_site!r} needs a "
                        "sequence model with a memory readout; "
                        f"{self.head.seq_model.name!r} has none"
                    )
                self.head.expose_memory_embeds = True
            if self.aux_canon_site == "memory":
                _cpc_in = self.head.memory_embed_size
            elif self.aux_canon_site in ("memory_obs", "probe"):
                _cpc_in = self.head.encoded_obs_size + self.head.memory_embed_size
            else:
                _cpc_in = self.head.embedding_size
            self.aux_cpc_head = AuxCanonCPC(
                embed_size=_cpc_in,
                label_size=label_width(
                    self.aux_canon_use_stone,
                    self.aux_canon_use_potion,
                    self.aux_canon_use_graph,
                ),
                proj_dim=int(getattr(config_rl, "aux_cpc_proj_dim", 128)),
                tau=float(getattr(config_rl, "aux_cpc_tau", 0.1)),
            )

        aux_head_parameters = (
            (
                tuple(self.aux_cpc_head.parameters())
                if self.aux_cpc_head is not None
                else ()
            )
            + (
                tuple(self.aux_canon_head.parameters())
                if self.aux_canon_head is not None
                else ()
            )
            + (
                tuple(self.aux_count_head.parameters())
                if self.aux_count_head is not None
                else ()
            )
        )

        # Optimizer
        if self.alternating_msc:
            self._rl_parameters = (
                *self.head.rl_parameters(),
                *self.qf.parameters(),
                *aux_head_parameters,
            )
            self._msc_parameters = tuple(self.head.msc_parameters())
            if not self._rl_parameters:
                raise ValueError("Alternating MSC requires RL parameters")
            if not self._msc_parameters:
                raise ValueError("Alternating MSC requires MSC parameters")
            if not {
                id(param) for param in self._rl_parameters
            }.isdisjoint(id(param) for param in self._msc_parameters):
                raise ValueError(
                    "Alternating MSC RL and MSC parameter lists must be disjoint"
                )
        else:
            self._rl_parameters = (
                *self.head.parameters(),
                *self.qf.parameters(),
                *aux_head_parameters,
            )
            self._msc_parameters = ()

        self.critic_optimizer = AdamW(
            self._rl_parameters,
            lr=config_rl.critic_lr,
            weight_decay=0.001,
        )
        # reference to https://github.com/UT-Austin-RPL/amago/blob/main/amago/experiment.py
        self.lr_schedule = get_constant_schedule_with_warmup(
            optimizer=self.critic_optimizer, num_warmup_steps=500
        )
        if self.alternating_msc:
            msc_lr = float(
                config_seq.seq_model.get("msc_lr", config_rl.critic_lr)
            )
            if msc_lr <= 0.0:
                raise ValueError(
                    "config_seq.seq_model.msc_lr must be positive"
                )
            self.aux_optimizer = AdamW(
                self._msc_parameters,
                lr=msc_lr,
                weight_decay=0.001,
            )
            self.aux_lr_schedule = get_constant_schedule_with_warmup(
                optimizer=self.aux_optimizer,
                num_warmup_steps=500,
            )

    def _build_qf(self, config_rl, config_env):
        """Flat critic, or the factored Alchemy head when asked for."""
        hidden_sizes = config_rl.config_critic.hidden_dims
        # aux_canon_feed_critic appends the decoded potion posterior.
        critic_in = self.head.embedding_size + self.aux_canon_feed_dim
        if not bool(getattr(config_rl, "factored_action_head", False)):
            return FlattenMlp(
                input_size=critic_in,
                output_size=self.action_dim,
                hidden_sizes=hidden_sizes,
            )
        if getattr(config_env, "env_type", None) != "alchemy":
            raise ValueError(
                "factored_action_head assumes the Alchemy "
                "NO_OP + stones x targets action layout"
            )
        layout = get_symbolic_alchemy_layout(
            bool(getattr(config_env, "observe_used", True))
        )
        head = FactoredAlchemyQHead(
            input_size=critic_in,
            hidden_sizes=hidden_sizes,
            max_stones=layout.max_stones,
            targets_per_stone=layout.potions_per_stone,
        )
        if head.action_dim != self.action_dim:
            raise ValueError(
                f"Factored head spans {head.action_dim} actions but the env "
                f"exposes {self.action_dim}"
            )
        return head

    def _strip_aux_target(self, observs):
        """Excise the 21-dim aux TARGET block from a raw observation.

        THIS IS THE LEAK GUARD. Every path that hands an observation to the
        network (`act`, `sample_random_action`, `_compute_loss`) goes through
        here first, so `RNN_head` / the critic / the action mask only ever see
        `net_obs_dim` features and can never read the label they are trained
        to predict.
        """
        if not self.aux_canon_target:
            return observs
        assert observs.shape[-1] == self.obs_dim, (
            f"expected raw obs width {self.obs_dim}, got {observs.shape[-1]}"
        )
        stripped = torch.cat(
            (
                observs[..., : self._aux_start],
                observs[..., self._aux_end:],
            ),
            dim=-1,
        )
        assert stripped.shape[-1] == self.net_obs_dim
        return stripped

    def _aux_target_slice(self, observs):
        """The raw 21-dim aux TARGET block (labels only, never an input)."""
        assert observs.shape[-1] == self.obs_dim
        return observs[..., self._aux_start:self._aux_end]

    @torch.no_grad()
    def act(
        self,
        prev_internal_state,
        prev_action,
        prev_reward,
        prev_obs,
        obs,
        deterministic=False,
        initial=False,
        timestep=0,
        skip_memory_update=False,
    ):
        prev_action = prev_action.unsqueeze(0)  # (1, B, dim)
        prev_reward = prev_reward.unsqueeze(0)  # (1, B, 1)
        prev_obs = self._strip_aux_target(prev_obs).unsqueeze(0)  # (1, B, dim)
        raw_obs = self._strip_aux_target(obs)
        obs = raw_obs.unsqueeze(0)              # (1, B, dim)

        step_out = self.head.step(
            prev_internal_state=prev_internal_state,
            prev_action=prev_action,
            prev_reward=prev_reward,
            prev_obs=prev_obs,
            obs=obs,
            initial=initial,
            timestep=timestep,
            skip_memory_update=skip_memory_update,
            return_parts=self.aux_canon_feed_critic,
        )
        if self.aux_canon_feed_critic:
            joint_embed, current_internal_state, step_obs, step_mem = step_out
            # The critic was TRAINED on [joint ++ decoded posterior]; acting on
            # the bare joint embedding would be a different network.
            joint_embed = torch.cat(
                (joint_embed,
                 self._decoded_potion_posterior(step_obs, step_mem)),
                dim=-1,
            )
        else:
            joint_embed, current_internal_state = step_out

        current_action = self._select_action(
            joint_embed,
            deterministic,
            raw_obs=raw_obs,
        )

        return current_action, current_internal_state

    def _valid_action_mask(self, raw_obs: torch.Tensor | None):
        if not self.mask_alchemy_invalid_actions or raw_obs is None:
            return None
        mask = valid_action_mask_from_observation(
            raw_obs,
            **self._alchemy_mask_kwargs,
        )
        if mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"Alchemy action mask width {mask.shape[-1]} does not match "
                f"action_dim {self.action_dim}"
            )
        return mask

    @staticmethod
    def _masked_argmax(
        action_logits: torch.Tensor,
        valid_action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if valid_action_mask is None:
            return torch.argmax(action_logits, dim=-1)
        min_value = torch.finfo(action_logits.dtype).min
        masked_logits = action_logits.masked_fill(
            ~valid_action_mask,
            min_value,
        )
        return torch.argmax(masked_logits, dim=-1)

    @staticmethod
    def _sample_valid_random_actions(
        batch_shape,
        valid_action_mask: torch.Tensor | None,
        *,
        action_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_action_mask is None:
            return torch.randint(
                high=action_dim,
                size=batch_shape,
                device=device,
            )
        random_scores = torch.rand(
            (*batch_shape, action_dim),
            device=device,
        )
        random_scores = random_scores.masked_fill(
            ~valid_action_mask,
            -1.0,
        )
        return torch.argmax(random_scores, dim=-1)

    def sample_random_action(
        self,
        *,
        raw_obs: torch.Tensor | None = None,
        batch_shape=None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Public entry (the Learner calls this with the RAW observation)."""
        if raw_obs is not None:
            raw_obs = self._strip_aux_target(raw_obs)
        return self._sample_random_action_net(
            raw_obs=raw_obs,
            batch_shape=batch_shape,
            device=device,
        )

    def _sample_random_action_net(
        self,
        *,
        raw_obs: torch.Tensor | None = None,
        batch_shape=None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Same, but `raw_obs` is already aux-stripped."""
        if raw_obs is not None:
            batch_shape = raw_obs.shape[:-1]
            device = raw_obs.device
        if batch_shape is None:
            raise ValueError(
                "sample_random_action requires raw_obs or batch_shape"
            )
        if device is None:
            device = ptu.device
        valid_action_mask = self._valid_action_mask(raw_obs)
        action = self._sample_valid_random_actions(
            batch_shape=batch_shape,
            valid_action_mask=valid_action_mask,
            action_dim=self.action_dim,
            device=device,
        )
        return F.one_hot(
            action.long(),
            num_classes=self.action_dim,
        ).float()

    def _select_action(
        self,
        observ,
        deterministic: bool,
        *,
        raw_obs: torch.Tensor | None = None,
    ):
        batch_size = observ.shape[0]
        action_logits = self.qf(observ)
        valid_action_mask = self._valid_action_mask(raw_obs)
        if deterministic:
            action = self._masked_argmax(
                action_logits,
                valid_action_mask,
            )
        else:
            random_action = torch.argmax(
                self._sample_random_action_net(
                    raw_obs=raw_obs,
                    batch_shape=action_logits.shape[:-1],
                    device=action_logits.device,
                ),
                dim=-1,
            )
            optimal_action = self._masked_argmax(
                action_logits,
                valid_action_mask,
            )

            eps = self.epsilon_schedule(self.count)
            mask = torch.multinomial(
                input=ptu.FloatTensor([1 - eps, eps]),
                num_samples=action_logits.shape[0],
                replacement=True,
            )
            action = mask * random_action + (1 - mask) * optimal_action
            self.count += batch_size

        return F.one_hot(
            action.long(),
            num_classes=action_logits.shape[-1],
        ).float()

    def _compute_loss(
        self, actions, rewards, observs, terms, masks, pos_offset=None,
        memory_mask=None,
    ):
        """
        actions[t] = a_{t-1}, shape (T+1, B, A)   one-hot
        rewards[t] = r_{t-1}, shape (T+1, B, 1)
        observs[t] = o_{t-1}, shape (T+2, B, dim)
        terms[t]   = done_{t-1}, shape (T+1, B, 1)
        masks[t]   = mask_{t-1}, shape (T+1, B, 1)
        """
        assert actions.dim() == rewards.dim() == terms.dim() == observs.dim() == masks.dim() == 3
        assert actions.shape[0] == rewards.shape[0] == terms.shape[0] == observs.shape[0] - 1 == masks.shape[0]
        # Peel the supervision labels off FIRST, then strip them: nothing below
        # this line ever sees the aux block.
        aux_canon_targets = (
            self._aux_target_slice(observs)
            if (self.aux_canon_enabled or self.aux_cpc_enabled) else None
        )
        observs = self._strip_aux_target(observs)
        loss_mask = masks
        if self.mask_rl_loss_on_reset_transition and memory_mask is not None:
            loss_mask = masks * memory_mask

        ### 1. Compute embeddings once
        joint_embeds, d_forward = self.head.forward(
            actions=actions, rewards=rewards, observs=observs, masks=masks,
            pos_offset=pos_offset, memory_mask=memory_mask,
        )  # (T+2, B, dim)
        # Popped here, not later: aux_canon_feed_critic consumes them before
        # the critic runs, and everything left in d_forward is logged as-is.
        memory_embeds = d_forward.pop("_memory_embeds", None)
        encoded_obs = d_forward.pop("_encoded_obs", None)
        obs_embeds = d_forward.pop("_obs_embeds", None)
        if self.aux_canon_feed_critic:
            if memory_embeds is None or encoded_obs is None:
                raise RuntimeError(
                    "aux_canon_feed_critic needs RNN_head to expose "
                    "_encoded_obs/_memory_embeds; expose_memory_embeds unset"
                )
            fed = self._decoded_potion_posterior(encoded_obs, memory_embeds)
            joint_embeds = torch.cat((joint_embeds, fed), dim=-1)
        target_joint_embeds = joint_embeds.detach()
        ### 2. Critic loss (DDQN)
        # Current Q values (raw / pre-POP-affine) — .detach() used for target computation below
        q_pred_all_raw = self.qf(joint_embeds)  # (T+2, B, A)
        next_valid_action_mask = self._valid_action_mask(observs[1:])

        with torch.no_grad():
            # DDQN: online net selects next action, target net evaluates its value
            next_actions = self._masked_argmax(
                q_pred_all_raw.detach()[1:],
                next_valid_action_mask,
            ).unsqueeze(-1)  # (T+1, B, 1)
            next_q_target_raw = self.qf_target(target_joint_embeds)[1:]  # (T+1, B, A)
            next_q_raw = next_q_target_raw.gather(-1, next_actions)  # (T+1, B, 1)
            next_q_denorm = self.popart(next_q_raw, normalized=False)  # denorm → reward scale
            q_target_denorm = rewards + (1.0 - terms) * self.gamma * next_q_denorm  # (T+1, B, 1) reward scale
            self.popart.update_stats(q_target_denorm, loss_mask)
            q_target_norm = self.popart.normalize_values(q_target_denorm)

        # Gather Q(h_t, a_t) from (T+1) slice — critic outputs raw Q (pre-POP-affine)
        actions_idx = torch.argmax(actions, dim=-1, keepdim=True)  # (T+1, B, 1)
        q_pred_raw = q_pred_all_raw[:-1].gather(-1, actions_idx)  # (T+1, B, 1)

        # Apply POP affine (w*x + b) before Bellman residual so stats shifts preserve gradient signal.
        q_pred_norm = self.popart(q_pred_raw)
        qf_elementwise = F.huber_loss(
            q_pred_norm,
            q_target_norm,
            reduction="none",
        )
        qf_elementwise = qf_elementwise * loss_mask
        num_valid_per_timestep = loss_mask.sum(dim=(1, 2)).clamp(min=1.0)
        qf_loss = qf_elementwise.sum(dim=(1, 2)) / num_valid_per_timestep
        num_valid = loss_mask.sum().clamp(min=1.0)
        critic_loss = qf_elementwise.sum() / num_valid

        # Denormalize for interpretable logging (critic outputs are raw / pre-affine)
        q_pred_denorm = self.popart(q_pred_raw, normalized=False)
        outputs = {
            "critic_loss": critic_loss.detach(),
            "qf_loss": qf_loss.detach(),
            "q": ((q_pred_denorm * loss_mask).sum() / num_valid).detach(),
            "target_q": ((q_target_denorm * loss_mask).sum() / num_valid).detach(),
        }
        # Seq-model aux loss (e.g. MSC; training-only); non-detached, so pop before logging.
        aux_loss = d_forward.pop("_aux_loss", None)
        # Popped unconditionally: it is a big non-detached tensor and must
        # never survive into outputs, which the Learner logs.
        if self.alternating_msc and aux_loss is not None:
            raise RuntimeError(
                "Alternating MSC RL forward unexpectedly returned _aux_loss; "
                "MSC loss must be optimized through update_msc()"
            )
        outputs.update(d_forward)

        ### 3. Update
        total_loss = critic_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss
            outputs["aux_loss"] = aux_loss.detach()

        ### 3b. Auxiliary canonical-frame supervision.
        # site="joint":      on the critic's own input (shared trunk).
        # site="memory":     on the memory readout h_t alone.
        # site="memory_obs": on cat(encoded_obs.detach(), h_t) -- the head can
        #                    READ the current frame but sends no gradient into
        #                    it, so only the memory is shaped.
        # site="probe":      memory_obs with the MEMORY detached too, so the
        #                    head trains but the agent does not -- a pure
        #                    measurement of what the memory already holds.
        if self.aux_canon_enabled:
            if self.aux_canon_site in ("memory", "memory_obs", "probe"):
                if memory_embeds is None:
                    raise RuntimeError(
                        f"aux_canon_site={self.aux_canon_site!r} but RNN_head "
                        "did not return _memory_embeds; expose_memory_embeds "
                        "was not set"
                    )
                if self.aux_canon_site == "memory":
                    aux_embeds = memory_embeds
                else:
                    if encoded_obs is None:
                        raise RuntimeError(
                            f"aux_canon_site={self.aux_canon_site!r} but "
                            "RNN_head did not return _encoded_obs"
                        )
                    # The detach IS the experiment. Both tensors are (T+2, B, ·)
                    # and share the same time alignment, so concatenating on the
                    # feature axis keeps `_aux_canon_loss`'s slicing valid.
                    memory_in = (
                        memory_embeds.detach()
                        if self.aux_canon_site == "probe"
                        else memory_embeds
                    )
                    aux_embeds = torch.cat(
                        (encoded_obs.detach(), memory_in), dim=-1
                    )
            else:
                aux_embeds = joint_embeds
            aux_canon_loss, aux_canon_metrics = self._aux_canon_loss(
                aux_embeds,
                observs,
                aux_canon_targets,
                loss_mask,
            )
            total_loss = total_loss + self.aux_canon_weight * aux_canon_loss
            outputs.update(aux_canon_metrics)

        ### 3b-ii. The same label, contrastive instead of regressed.
        if self.aux_cpc_enabled:
            cpc_embeds = self._aux_site_embeds(
                joint_embeds, memory_embeds, encoded_obs
            )
            cpc_loss, cpc_metrics = self.aux_cpc_loss(
                cpc_embeds, aux_canon_targets, loss_mask
            )
            total_loss = total_loss + self.aux_cpc_weight * cpc_loss
            outputs.update(cpc_metrics)

        ### 3c. "Predict: Features" counting loss. See AUX_COUNT_SITES for
        # which representation the head reads; "obs" is the original site and
        # "memory"/"memory_obs" are the paper-faithful ones.
        if self.aux_count_enabled:
            count_embeds = self._aux_count_site_embeds(
                obs_embeds, memory_embeds, encoded_obs
            )
            aux_count_loss, aux_count_metrics = self._aux_count_loss(
                count_embeds, observs, loss_mask
            )
            total_loss = total_loss + self.aux_count_weight * aux_count_loss
            outputs.update(aux_count_metrics)

        return total_loss, outputs

    def _aux_count_site_embeds(self, obs_embeds, memory_embeds, encoded_obs):
        """Where the counting head reads from. See AUX_COUNT_SITES."""
        if self.aux_count_site == "obs":
            if obs_embeds is None:
                raise RuntimeError(
                    "aux_count_weight > 0 but RNN_head did not return "
                    "_obs_embeds; expose_obs_embeds was not set"
                )
            return obs_embeds
        if memory_embeds is None:
            raise RuntimeError(
                f"aux_count_site={self.aux_count_site!r} but RNN_head did not "
                "return _memory_embeds; expose_memory_embeds was not set"
            )
        if self.aux_count_site == "memory":
            return memory_embeds
        if encoded_obs is None:
            raise RuntimeError(
                f"aux_count_site={self.aux_count_site!r} but RNN_head did not "
                "return _encoded_obs"
            )
        # The detach IS the experiment: "probe" trains the head but not the
        # agent, so the metric reports what the memory already holds.
        memory_in = (
            memory_embeds.detach()
            if self.aux_count_site == "probe"
            else memory_embeds
        )
        return torch.cat((encoded_obs.detach(), memory_in), dim=-1)

    def _aux_count_loss(self, obs_embeds, observs, loss_mask):
        """Masked MSE on the two "Predict: Features" count vectors.

        `obs_embeds` is the conditioner's observation branch, (T+2, B, ·),
        sharing the time alignment every other term here uses: entry t of the
        first T+1 rows lines up with `loss_mask[t]`.

        Targets come from `count_targets_from_observation`, a function of the
        observation the agent is already holding -- no new information, and the
        same slot-occupancy flags the action mask reads. What the loss buys is
        the SHAPE of the computation: producing a per-category count requires
        reading every slot the same way and summing, which is the
        permutation-invariant structure a flat MLP over concatenated slots does
        not otherwise acquire.

        Metrics stay on the GPU (see CLAUDE.md: no `.item()`, no `.cpu()`).
        """
        embeds = obs_embeds[:-1]                          # (T+1, B, dim)
        stone_counts, potion_counts = count_targets_from_observation(
            observs[:-1], **self._alchemy_split_kwargs
        )
        denom = loss_mask.sum().clamp(min=1.0)

        out = self.aux_count_head(embeds)
        cursor = 0
        aux_loss = torch.zeros((), device=embeds.device, dtype=embeds.dtype)
        metrics = {}

        for use, width, target, name in (
            (self.aux_count_use_stone, AUX_COUNT_STONE_OUT, stone_counts, "stone"),
            (self.aux_count_use_potion, AUX_COUNT_POTION_OUT, potion_counts, "potion"),
        ):
            if not use:
                continue
            pred = out[..., cursor:cursor + width]
            cursor += width
            se = ((pred - target) ** 2).mean(dim=-1, keepdim=True)
            loss = (se * loss_mask).sum() / denom
            aux_loss = aux_loss + loss
            metrics[f"aux_count_{name}_loss"] = loss.detach()
            with torch.no_grad():
                # Rounded-count error, in raw units of "slots miscounted".
                ae = (pred.round() - target).abs().sum(dim=-1, keepdim=True)
                metrics[f"aux_count_{name}_mae"] = (
                    (ae * loss_mask).sum() / denom
                )

        metrics["aux_count_loss"] = aux_loss.detach()
        return aux_loss, metrics

    def _add_per_trial_metrics(self, metrics, name, hit, mask):
        """Split a (T+1, B, S) per-step accuracy into one scalar per trial.

        The averaged accuracy answers "how much of the chemistry does the
        memory hold", but not "does it ACCUMULATE" -- and only the second is
        meta-learning. A flat 0.55 at every trial and a 0.30 -> 0.80 climb
        report the same average and imply opposite next experiments.

        Row 0 is the dummy step at t=-1 (mask 0) and rows 1..T are env steps
        0..T-1, so trial i owns rows [1 + i*L, 1 + (i+1)*L). If the episode
        does not factor that way -- a truncated run, or a non-native
        multi-attempt wrapper -- the split is skipped rather than reported
        wrong. All values stay 0-dim GPU tensors (CLAUDE.md: no sync).
        """
        n_trials = self._alchemy_num_trials
        length = self._alchemy_steps_per_trial
        if n_trials <= 1 or length <= 0:
            return
        if hit.shape[0] != n_trials * length + 1:
            return
        per_step = (hit * mask)[1:].reshape(n_trials, length, *hit.shape[1:])
        per_step_mask = mask[1:].reshape(n_trials, length, *mask.shape[1:])
        num = per_step.flatten(1).sum(dim=-1)
        den = per_step_mask.flatten(1).sum(dim=-1).clamp(min=1.0)
        per_trial = num / den
        for i in range(n_trials):
            metrics[f"{name}_trial{i}"] = per_trial[i]

    def _decoded_potion_posterior(self, encoded_obs, memory_embeds):
        """Detached (..., 12*6) softmax posterior over latent potion types.

        Detached on purpose: the decoder is trained by the auxiliary loss ONLY.
        If the critic's gradient reached it, a critic that found the decode
        inconvenient could quietly degrade it, and the experiment would no
        longer be "does having the decode help".
        """
        aux_in = torch.cat(
            (encoded_obs.detach(), memory_embeds.detach()), dim=-1
        )
        out = self.aux_canon_head(aux_in)
        # potion logits sit after the stone block when both halves are on
        start = AUX_CANON_STONE_OUT * int(self.aux_canon_use_stone)
        logits = out[..., start:start + AUX_CANON_POTION_OUT].reshape(
            *out.shape[:-1], AUX_CANON_POTION_DIM, AUX_CANON_NUM_POTION_TYPES
        )
        return torch.softmax(logits, dim=-1).flatten(-2).detach()

    def _aux_site_embeds(self, joint_embeds, memory_embeds, encoded_obs):
        """Where an aux head reads from. See the site comment above `_aux_canon_loss`."""
        if self.aux_canon_site == "joint":
            return joint_embeds
        if memory_embeds is None:
            raise RuntimeError(
                f"aux_canon_site={self.aux_canon_site!r} but RNN_head did not "
                "return _memory_embeds; expose_memory_embeds was not set"
            )
        if self.aux_canon_site == "memory":
            return memory_embeds
        if encoded_obs is None:
            raise RuntimeError(
                f"aux_canon_site={self.aux_canon_site!r} but RNN_head did not "
                "return _encoded_obs"
            )
        memory_in = (
            memory_embeds.detach()
            if self.aux_canon_site == "probe"
            else memory_embeds
        )
        return torch.cat((encoded_obs.detach(), memory_in), dim=-1)

    def aux_cpc_loss(self, aux_embeds, targets, loss_mask):
        """InfoNCE between the memory read-out and the canonical label."""
        labels = encode_label(
            targets[:-1],
            self.aux_canon_use_stone,
            self.aux_canon_use_potion,
            self.aux_canon_use_graph,
        )
        return self.aux_cpc_head(aux_embeds[:-1], labels, loss_mask)

    def _aux_canon_loss(self, aux_embeds, observs, targets, loss_mask):
        """Masked MSE on latent stone coords + masked CE on latent potion types.

        `aux_embeds` is whatever `aux_canon_site` selected — the critic's joint
        embedding or the memory readout h_t. Both are (T+2, B, ·) and share the
        same time alignment, so the slicing below is identical either way.

        `aux_embeds` / `observs` are (T+2, B, ·) and `loss_mask` is
        (T+1, B, 1); index t of the first T+1 entries lines up with
        `loss_mask[t]`, exactly as the critic term above uses them. Slots are
        additionally masked by the present flags read off the observation's own
        used-flags — the same flags `valid_action_mask_from_observation` reads,
        so the two can never disagree. Absent slots carry an out-of-band
        sentinel in the target and contribute nothing.

        All returned metrics stay on the GPU (see CLAUDE.md: no `.item()`,
        no `.cpu()`, no python-level branching on a tensor).
        """
        embeds = aux_embeds[:-1]                         # (T+1, B, dim)
        targets = targets[:-1]                           # (T+1, B, 21)
        lead = embeds.shape[:-1]

        stone_present, potion_present = present_flags_from_observation(
            observs[:-1],
            **self._alchemy_split_kwargs,
        )
        stone_mask = stone_present.to(embeds.dtype) * loss_mask  # (T+1,B,3)
        potion_mask = potion_present.to(embeds.dtype) * loss_mask  # (T+1,B,12)

        out = self.aux_canon_head(embeds)                # (T+1, B, out_dim)
        cursor = 0
        aux_loss = torch.zeros((), device=embeds.device, dtype=embeds.dtype)
        metrics = {}

        if self.aux_canon_use_stone:
            pred_coord = out[..., cursor:cursor + AUX_CANON_STONE_OUT].reshape(
                *lead, -1, 3
            )
            cursor += AUX_CANON_STONE_OUT
            tgt_coord = targets[..., :AUX_CANON_STONE_DIM].reshape(*lead, -1, 3)
            stone_denom = stone_mask.sum().clamp(min=1.0)
            coord_se = ((pred_coord - tgt_coord) ** 2).mean(dim=-1)  # (T+1,B,3)
            stone_loss = (coord_se * stone_mask).sum() / stone_denom
            aux_loss = aux_loss + stone_loss
            with torch.no_grad():
                coord_hit = (
                    (pred_coord > 0) == (tgt_coord > 0)
                ).to(embeds.dtype).mean(dim=-1)
                metrics["aux_canon_stone_acc"] = (
                    (coord_hit * stone_mask).sum() / stone_denom
                )
            metrics["aux_canon_stone_loss"] = stone_loss.detach()

        if self.aux_canon_use_potion:
            pred_type = out[..., cursor:cursor + AUX_CANON_POTION_OUT].reshape(
                *lead, AUX_CANON_POTION_DIM, AUX_CANON_NUM_POTION_TYPES
            )
            # Absent slots hold AUX_CANON_ABSENT; clamp keeps CE's index lookup
            # in range and the mask removes their contribution entirely.
            tgt_type = (
                targets[
                    ...,
                    AUX_CANON_STONE_DIM:AUX_CANON_STONE_DIM + AUX_CANON_POTION_DIM
                ]
                .long()
                .clamp(0, AUX_CANON_NUM_POTION_TYPES - 1)
            )
            potion_denom = potion_mask.sum().clamp(min=1.0)
            ce = F.cross_entropy(
                pred_type.reshape(-1, AUX_CANON_NUM_POTION_TYPES),
                tgt_type.reshape(-1),
                reduction="none",
            ).reshape(potion_mask.shape)
            potion_loss = (ce * potion_mask).sum() / potion_denom
            aux_loss = aux_loss + potion_loss
            with torch.no_grad():
                type_hit = (
                    pred_type.argmax(dim=-1) == tgt_type
                ).to(embeds.dtype)
                metrics["aux_canon_potion_acc"] = (
                    (type_hit * potion_mask).sum() / potion_denom
                )
                self._add_per_trial_metrics(
                    metrics, "aux_canon_potion_acc", type_hit, potion_mask
                )
            metrics["aux_canon_potion_loss"] = potion_loss.detach()

        if self.aux_canon_use_graph:
            pred_edge = out[..., cursor:cursor + AUX_CANON_GRAPH_OUT]
            cursor += AUX_CANON_GRAPH_OUT
            tgt_edge = targets[..., -AUX_CANON_GRAPH_DIM:]
            # No slot mask: unlike stones and potions, an edge is defined at
            # every step of the episode, so only the rollout mask applies.
            denom = loss_mask.sum().clamp(min=1.0) * AUX_CANON_GRAPH_DIM
            bce = F.binary_cross_entropy_with_logits(
                pred_edge, tgt_edge, reduction="none"
            )
            graph_loss = (bce * loss_mask).sum() / denom
            aux_loss = aux_loss + graph_loss
            metrics["aux_canon_graph_loss"] = graph_loss.detach()
            with torch.no_grad():
                hit = ((pred_edge > 0) == (tgt_edge > 0.5)).to(embeds.dtype)
                metrics["aux_canon_graph_acc"] = (
                    (hit * loss_mask).sum() / denom
                )
                self._add_per_trial_metrics(
                    metrics, "aux_canon_graph_acc", hit,
                    loss_mask.expand_as(hit),
                )

        metrics["aux_canon_loss"] = aux_loss.detach()
        return aux_loss, metrics

    def forward(
        self, actions, rewards, observs, terms, masks, pos_offset=None,
        memory_mask=None,
    ):
        compute_loss = self._compute_loss
        if self.compile_training_loss and actions.is_cuda:
            if self._compiled_compute_loss is None:
                self._compiled_compute_loss = torch.compile(
                    self._compute_loss,
                    dynamic=False,
                )
            compute_loss = self._compiled_compute_loss

        total_loss, outputs = compute_loss(
            actions,
            rewards,
            observs,
            terms,
            masks,
            pos_offset,
            memory_mask,
        )
        outputs.update(self.popart.metrics())

        self.critic_optimizer.zero_grad()
        total_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            outputs.update(
                clip_gradients(
                    self._rl_parameters,
                    self.clip_grad_norm,
                )
            )

        self.critic_optimizer.step()
        self.lr_schedule.step()

        ### 4. Soft update
        self.soft_target_update()

        return outputs

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf, self.qf_target, self.tau)

    def training_state_dict(self):
        state_dict = {
            "model": self.state_dict(),
            "optimizer": self.critic_optimizer.state_dict(),
            "lr_schedule": self.lr_schedule.state_dict(),
            "count": self.count,
        }
        if self.alternating_msc:
            state_dict.update(
                {
                    "aux_optimizer": self.aux_optimizer.state_dict(),
                    "aux_lr_schedule": self.aux_lr_schedule.state_dict(),
                }
            )
        return state_dict

    def load_training_state_dict(self, state_dict):
        self.load_state_dict(state_dict["model"])
        self.critic_optimizer.load_state_dict(state_dict["optimizer"])
        self.lr_schedule.load_state_dict(state_dict["lr_schedule"])
        if self.alternating_msc:
            self.aux_optimizer.load_state_dict(state_dict["aux_optimizer"])
            self.aux_lr_schedule.load_state_dict(
                state_dict["aux_lr_schedule"]
            )
        self.count = int(state_dict["count"])

    def update(self, batch):
        recurrent_batch = prepare_recurrent_batch(
            batch,
            discrete_action_dim=self.action_dim,
        )

        return self.forward(
            recurrent_batch.actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.terms,
            recurrent_batch.masks,
            recurrent_batch.pos_offset,
            recurrent_batch.memory_mask,
        )

    def update_msc(self, batch):
        if not self.alternating_msc:
            raise RuntimeError(
                "update_msc requires alternating_ema MSC mode"
            )

        recurrent_batch = prepare_recurrent_batch(
            batch,
            discrete_action_dim=self.action_dim,
        )
        raw_loss, outputs = self.head.compute_msc_loss(
            recurrent_batch.actions,
            recurrent_batch.rewards,
            recurrent_batch.observs,
            recurrent_batch.masks,
            recurrent_batch.memory_mask,
        )

        self.aux_optimizer.zero_grad()
        raw_loss.backward()

        if self.clip and self.clip_grad_norm > 0.0:
            grad_metrics = clip_gradients(
                self._msc_parameters,
                self.clip_grad_norm,
            )
            outputs.update(
                {
                    f"msc_{key}": value
                    for key, value in grad_metrics.items()
                }
            )

        self.aux_optimizer.step()
        self.aux_lr_schedule.step()
        self.head.update_msc_ema(self.tau)

        return outputs
