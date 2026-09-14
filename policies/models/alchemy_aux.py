"""Symbolic Alchemy auxiliary supervision and action masking, for main's agents.

WHY THIS FILE EXISTS. The Alchemy work was developed against an older RNN_head
whose batches carried a dummy row at t=-1 and were shaped (T+2, B, .). main
since reworked the replay path: every tensor an agent sees is now (L, B, .) and
aligned, with `observs[t] = s_{j_t-1}` and `actions[t] = a_{j_t-1}`. Re-deriving
the auxiliary losses against that alignment is shorter and less error-prone than
porting slices that were written for the old one, so the arithmetic lives here
rather than being threaded back through the agent.

WHAT THE AUXILIARY LOSS IS FOR. The DDQN oracle is handed the episode's true
chemistry and still scores 184.8 against a 173.5 floor, while a memoryless agent
on `all_fixed` -- where the per-episode frame map is constant -- reaches 302. The
information is there; what the scalar TD signal never drives is the computation
of frame_map(perceived_slot, chem_gt[12:28]). `scripts/probe_frame_map.py` shows
the same critic MLP fits that map to 100% test accuracy inside one epoch. This
loss makes it an explicit target, and takes the oracle to 297.8 (+0.877).

NOT AN INPUT. The env appends the label to the observation; the agent excises it
before RNN_head, the critic or the action mask can see it. `strip_target` and
`target_slice` are the only two functions that touch that block, so there is one
place to audit rather than several.

SITE. Which representation the head reads decides what the gradient shapes:
  joint       the critic's own input -- shared trunk, and the interference is
              real (measured -30.4 on MATE at weight 1)
  memory      the memory read-out alone. MIS-SPECIFIED and kept only as a
              control: the target says "which latent type is in slot j RIGHT
              NOW", which needs the current frame, so a head given only h_t
              cannot express it (measured accuracy 0.261 against chance 0.167)
  memory_obs  cat(obs.detach(), h_t): may READ the current frame but sends
              gradient only into the memory. Recovers 23.7 of joint's 27.8 lost
              points at the same accuracy
  probe       memory_obs with the memory detached too -- measurement only, so
              reading the metric cannot change what it measures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.alchemy import (
    AUX_CANON_DIM,
    AUX_CANON_GRAPH_DIM,
    AUX_CANON_NUM_POTION_TYPES,
    AUX_CANON_POTION_DIM,
    AUX_CANON_STONE_DIM,
    present_flags_from_observation,
    valid_action_mask_from_observation,
)
from torchkit.networks import FlattenMlp

SITES = ("joint", "memory", "memory_obs", "probe")
PARTS = ("both", "stone", "potion", "graph", "potion_graph", "all")
STONE_OUT = AUX_CANON_STONE_DIM
POTION_OUT = AUX_CANON_POTION_DIM * AUX_CANON_NUM_POTION_TYPES
GRAPH_OUT = AUX_CANON_GRAPH_DIM
_MASK_FILL = -1e8


def parts_flags(parts):
    if parts not in PARTS:
        raise ValueError(f"aux_canon_parts must be one of {PARTS}, got {parts!r}")
    return (
        parts in ("both", "stone", "all"),
        parts in ("both", "potion", "potion_graph", "all"),
        parts in ("graph", "potion_graph", "all"),
    )


def head_width(use_stone, use_potion, use_graph):
    return (
        STONE_OUT * int(use_stone)
        + POTION_OUT * int(use_potion)
        + GRAPH_OUT * int(use_graph)
    )


class AlchemyAux(nn.Module):
    """Owns the label block, the auxiliary heads, and the action mask."""

    def __init__(self, obs_dim, action_dim, config_rl, config_env, config_seq):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.is_alchemy = str(getattr(config_env, "env_type", "")) == "alchemy"

        # ---- label block -------------------------------------------------
        self.target_enabled = bool(
            getattr(config_env, "aux_canon_target", False)
        ) and self.is_alchemy
        self.canon_weight = float(getattr(config_rl, "aux_canon_weight", 0.0))
        self.cpc_weight = float(getattr(config_rl, "aux_cpc_weight", 0.0))
        if (self.canon_weight > 0 or self.cpc_weight > 0) and not self.target_enabled:
            raise ValueError(
                "aux_canon_weight / aux_cpc_weight > 0 require "
                "config_env.aux_canon_target=True"
            )
        # Width the NETWORK must be built for: smaller than obs_dim by the label.
        self.net_obs_dim = self.obs_dim - (AUX_CANON_DIM if self.target_enabled else 0)

        self.site = str(getattr(config_rl, "aux_canon_site", "joint"))
        if self.site not in SITES:
            raise ValueError(f"aux_canon_site must be one of {SITES}, got {self.site!r}")
        self.use_stone, self.use_potion, self.use_graph = parts_flags(
            str(getattr(config_rl, "aux_canon_parts", "both"))
        )

        # ---- action mask -------------------------------------------------
        self.mask_invalid = bool(
            getattr(config_rl, "mask_alchemy_invalid_actions", False)
        ) and self.is_alchemy
        self._mask_kwargs = None
        if self.mask_invalid or self.canon_weight > 0 or self.cpc_weight > 0:
            self._mask_kwargs = dict(
                observe_used=bool(config_env.observe_used),
                add_trial_flag=bool(config_env.add_trial_flag),
                context_dim=int(getattr(config_seq.seq_model, "context_dim", 0)),
                structured_potions=bool(getattr(config_env, "structured_potions", False)),
                add_trial_phase=bool(getattr(config_env, "add_trial_phase", False)),
                aux_canon_target=False,   # always called on the STRIPPED obs
            )
        self.mask_no_op = bool(getattr(config_rl, "mask_alchemy_no_op", False))
        if self.mask_no_op and not self.mask_invalid:
            raise ValueError(
                "mask_alchemy_no_op=True requires mask_alchemy_invalid_actions=True"
            )

        self.canon_head = None
        self.cpc = None
        self.enabled = self.canon_weight > 0.0 or self.cpc_weight > 0.0
        self._config_rl = config_rl

    def build_heads(self, head):
        """Build the auxiliary heads once `head` exists.

        Split from __init__ because the agent needs `net_obs_dim` to size
        RNN_head, and the heads need RNN_head's widths -- a two-phase
        construction rather than a guess at either end.
        """
        if not self.enabled:
            return
        config_rl = self._config_rl
        if self.site == "joint":
            in_dim = head.embedding_size
        else:
            if head.memory_embed_size <= 0:
                raise ValueError(
                    f"aux_canon_site={self.site!r} needs a sequence model with a "
                    f"memory; got memory_embed_size={head.memory_embed_size}"
                )
            head.expose_memory_embeds = True
            in_dim = head.memory_embed_size
            if self.site in ("memory_obs", "probe"):
                in_dim += head.encoded_obs_size
        self.in_dim = in_dim
        if self.canon_weight > 0.0:
            width = head_width(self.use_stone, self.use_potion, self.use_graph)
            self.canon_head = FlattenMlp(
                input_size=in_dim, output_size=width,
                hidden_sizes=config_rl.config_critic.hidden_dims,
            )
        if self.cpc_weight > 0.0:
            from policies.models.aux_cpc import AuxCanonCPC

            self.cpc = AuxCanonCPC(
                embed_dim=in_dim,
                use_stone=self.use_stone,
                use_potion=self.use_potion,
                use_graph=self.use_graph,
                proj_dim=int(getattr(config_rl, "aux_cpc_proj_dim", 128)),
                tau=float(getattr(config_rl, "aux_cpc_tau", 0.1)),
            )

    # ---- label block, the only two places that touch it ------------------
    def strip_target(self, observs):
        if not self.target_enabled or observs is None:
            return observs
        return observs[..., : self.net_obs_dim]

    def target_slice(self, observs):
        if not self.target_enabled:
            return None
        return observs[..., self.net_obs_dim :]

    # ---- action mask ------------------------------------------------------
    def valid_action_mask(self, stripped_obs):
        """`stripped_obs` must already have the label block removed."""
        if not self.mask_invalid or stripped_obs is None:
            return None
        mask = valid_action_mask_from_observation(
            stripped_obs, mask_no_op=self.mask_no_op, **self._mask_kwargs
        )
        if mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"Alchemy action mask width {mask.shape[-1]} != action_dim "
                f"{self.action_dim}"
            )
        return mask

    @staticmethod
    def mask_logits(logits, mask):
        if mask is None:
            return logits
        # A row with no legal action would make an all -inf softmax/argmax
        # meaningless. The env always leaves NO_OP legal, but the fallback keeps
        # a malformed mask from poisoning a batch.
        empty = ~mask.any(dim=-1, keepdim=True)
        return logits.masked_fill(~(mask | empty), _MASK_FILL)

    def random_action(self, stripped_obs):
        """Uniform over LEGAL actions -- the warm-up would otherwise be no-ops."""
        mask = self.valid_action_mask(stripped_obs)
        if mask is None:
            return None
        # Gumbel-max over a uniform masked distribution: one draw per row
        # without building a Categorical per step.
        scores = torch.rand(mask.shape, device=mask.device).log()
        idx = self.mask_logits(scores, mask).argmax(dim=-1)
        return F.one_hot(idx, self.action_dim).float()

    # ---- site dispatch ----------------------------------------------------
    def site_embeds(self, joint, memory, encoded_obs):
        if self.site == "joint":
            return joint
        if memory is None:
            raise RuntimeError(
                f"aux_canon_site={self.site!r} but RNN_head returned no "
                "_memory_embeds; expose_memory_embeds was not set"
            )
        if self.site == "memory":
            return memory
        if encoded_obs is None:
            raise RuntimeError(
                f"aux_canon_site={self.site!r} but RNN_head returned no "
                "_encoded_obs"
            )
        # The detach IS the experiment: "probe" trains the head but not the
        # agent, so reading the metric cannot change what it measures.
        mem = memory.detach() if self.site == "probe" else memory
        return torch.cat((encoded_obs.detach(), mem), dim=-1)

    # ---- losses -----------------------------------------------------------
    def loss(self, embeds, stripped_obs, targets, masks):
        """embeds / stripped_obs / targets / masks are all (L, B, .) aligned.

        Slots are masked by the observation's own used-flags -- the same flags
        the action mask reads, so the two can never disagree. Absent slots carry
        an out-of-band sentinel in the target and contribute nothing.

        Metrics stay on the GPU (CLAUDE.md: no .item(), no .cpu()).
        """
        total = torch.zeros((), device=embeds.device, dtype=embeds.dtype)
        metrics = {}
        stone_present, potion_present = present_flags_from_observation(
            stripped_obs, **self._mask_kwargs
        )
        stone_mask = stone_present.to(embeds.dtype) * masks
        potion_mask = potion_present.to(embeds.dtype) * masks

        if self.canon_head is not None:
            out = self.canon_head(embeds)
            cursor = 0
            lead = embeds.shape[:-1]
            if self.use_stone:
                pred = out[..., cursor:cursor + STONE_OUT].reshape(*lead, -1, 3)
                cursor += STONE_OUT
                tgt = targets[..., :AUX_CANON_STONE_DIM].reshape(*lead, -1, 3)
                denom = stone_mask.sum().clamp(min=1.0)
                se = ((pred - tgt) ** 2).mean(dim=-1)
                stone_loss = (se * stone_mask).sum() / denom
                total = total + stone_loss
                metrics["aux_canon_stone_loss"] = stone_loss.detach()
                with torch.no_grad():
                    hit = ((pred > 0) == (tgt > 0)).to(embeds.dtype).mean(dim=-1)
                    metrics["aux_canon_stone_acc"] = (
                        (hit * stone_mask).sum() / denom
                    )
            if self.use_potion:
                pred = out[..., cursor:cursor + POTION_OUT].reshape(
                    *lead, AUX_CANON_POTION_DIM, AUX_CANON_NUM_POTION_TYPES
                )
                cursor += POTION_OUT
                # Absent slots hold the sentinel; clamp keeps CE's index lookup
                # in range and the mask removes their contribution entirely.
                tgt = targets[
                    ..., AUX_CANON_STONE_DIM:AUX_CANON_STONE_DIM + AUX_CANON_POTION_DIM
                ].long().clamp(0, AUX_CANON_NUM_POTION_TYPES - 1)
                denom = potion_mask.sum().clamp(min=1.0)
                ce = F.cross_entropy(
                    pred.reshape(-1, AUX_CANON_NUM_POTION_TYPES),
                    tgt.reshape(-1), reduction="none",
                ).reshape(potion_mask.shape)
                potion_loss = (ce * potion_mask).sum() / denom
                total = total + potion_loss
                metrics["aux_canon_potion_loss"] = potion_loss.detach()
                with torch.no_grad():
                    hit = (pred.argmax(dim=-1) == tgt).to(embeds.dtype)
                    metrics["aux_canon_potion_acc"] = (
                        (hit * potion_mask).sum() / denom
                    )
            if self.use_graph:
                pred = out[..., cursor:cursor + GRAPH_OUT]
                tgt = targets[..., -AUX_CANON_GRAPH_DIM:]
                # An edge is defined at every step, so only the rollout mask.
                denom = masks.sum().clamp(min=1.0) * AUX_CANON_GRAPH_DIM
                bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
                graph_loss = (bce * masks).sum() / denom
                total = total + graph_loss
                metrics["aux_canon_graph_loss"] = graph_loss.detach()
                with torch.no_grad():
                    hit = ((pred > 0) == (tgt > 0.5)).to(embeds.dtype)
                    metrics["aux_canon_graph_acc"] = (hit * masks).sum() / denom
            total = self.canon_weight * total

        if self.cpc is not None:
            cpc_loss, cpc_metrics = self.cpc(embeds, targets, masks)
            total = total + self.cpc_weight * cpc_loss
            metrics.update(cpc_metrics)

        metrics["aux_canon_loss"] = total.detach()
        return total, metrics
