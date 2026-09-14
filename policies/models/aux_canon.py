"""Canonical-frame auxiliary supervision, shared by the SAC agent.

`policy_rnn_dqn` grew this machinery inline. Rather than copy it -- the risky
part is the LEAK GUARD, and two copies of a leak guard is how one of them
silently rots -- the logic lives here and `scripts/verify_aux_canon_parity.py`
asserts that this module and the DQN implementation agree, index for index, on
the same config. The DQN class keeps its own copy until that parity check has
ridden along with a few runs; then it should be deleted in favour of this.

What this owns:
  * where the 33-dim label block sits inside the observation, and cutting it
    out before the network can ever read it;
  * the auxiliary head and its site dispatch (joint / memory / memory_obs /
    probe);
  * the masked losses and their metrics, including the per-trial split.
"""
import torch
from torch.nn import functional as F

from envs.alchemy import (
    AUX_CANON_DIM,
    AUX_CANON_GRAPH_DIM,
    AUX_CANON_NUM_POTION_TYPES,
    AUX_CANON_POTION_DIM,
    AUX_CANON_STONE_DIM,
    TRIAL_PHASE_DIM,
    get_symbolic_alchemy_layout,
    present_flags_from_observation,
)
from policies.models.aux_cpc import AuxCanonCPC, encode_label, label_width
from torchkit.networks import FlattenMlp

AUX_CANON_STONE_OUT = AUX_CANON_STONE_DIM
AUX_CANON_POTION_OUT = AUX_CANON_POTION_DIM * AUX_CANON_NUM_POTION_TYPES
AUX_CANON_GRAPH_OUT = AUX_CANON_GRAPH_DIM
AUX_CANON_PARTS = ("both", "stone", "potion", "graph", "potion_graph", "all")
AUX_CANON_SITES = ("joint", "memory", "memory_obs", "probe")


class AuxCanonMixin:
    """Mix into an agent that owns `self.head` (an RNN_head) and `self.obs_dim`."""

    def configure_aux_canon(self, config_rl, config_env, is_alchemy):
        """Set every aux flag, the stripped observation width, and the label slice.

        Call BEFORE building the network: `net_obs_dim` is what the network must
        be sized for, and it is smaller than `obs_dim` exactly by the label block.
        """
        self.aux_canon_target = bool(
            getattr(config_env, "aux_canon_target", False) and is_alchemy
        )
        self.aux_canon_weight = float(getattr(config_rl, "aux_canon_weight", 0.0))
        if self.aux_canon_weight > 0.0 and not self.aux_canon_target:
            raise ValueError(
                "config_rl.aux_canon_weight > 0 requires "
                "config_env.aux_canon_target=True"
            )
        self.aux_canon_enabled = (
            self.aux_canon_target and self.aux_canon_weight > 0.0
        )
        # Contrastive variant of the same label. Independent weight, so the two
        # can run alone or together.
        self.aux_cpc_weight = float(getattr(config_rl, "aux_cpc_weight", 0.0))
        if self.aux_cpc_weight > 0.0 and not self.aux_canon_target:
            raise ValueError(
                "config_rl.aux_cpc_weight > 0 requires "
                "config_env.aux_canon_target=True"
            )
        self.aux_cpc_enabled = self.aux_cpc_weight > 0.0
        self.aux_canon_parts = str(getattr(config_rl, "aux_canon_parts", "both"))
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
        self.aux_canon_site = str(getattr(config_rl, "aux_canon_site", "joint"))
        if self.aux_canon_site not in AUX_CANON_SITES:
            raise ValueError(
                f"config_rl.aux_canon_site must be one of {AUX_CANON_SITES}, "
                f"got {self.aux_canon_site!r}"
            )

        self.aux_any_enabled = self.aux_canon_enabled or self.aux_cpc_enabled
        self.net_obs_dim = self.obs_dim - (
            AUX_CANON_DIM if self.aux_canon_target else 0
        )
        self._aux_start = 0
        self._aux_end = 0
        self._alchemy_split_kwargs = None
        self._alchemy_num_trials = int(getattr(config_env, "num_trials", 0) or 0)
        self._alchemy_steps_per_trial = int(
            getattr(config_env, "max_steps_per_trial", 0) or 0
        )
        if not is_alchemy:
            return

        observe_used = bool(getattr(config_env, "observe_used", True))
        add_trial_flag = bool(getattr(config_env, "add_trial_flag", False))
        structured_potions = bool(getattr(config_env, "structured_potions", False))
        add_trial_phase = bool(getattr(config_env, "add_trial_phase", False))
        layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
        symbolic_obs_dim = (
            layout.symbolic_obs_dim
            + int(add_trial_flag)
            + (TRIAL_PHASE_DIM if add_trial_phase else 0)
        )
        # The label block sits after the env's own observation and before the
        # oracle's chem_gt tail, so every context_dim-based slice still works.
        self._aux_start = symbolic_obs_dim
        self._aux_end = symbolic_obs_dim + AUX_CANON_DIM
        context_dim = self.net_obs_dim - symbolic_obs_dim
        if context_dim < 0:
            raise ValueError(
                f"Alchemy obs too narrow: obs_dim={self.obs_dim}, "
                f"net_obs_dim={self.net_obs_dim}, symbolic={symbolic_obs_dim}"
            )
        self._alchemy_split_kwargs = {
            "observe_used": observe_used,
            "add_trial_flag": add_trial_flag,
            "context_dim": context_dim,
            "structured_potions": structured_potions,
            "add_trial_phase": add_trial_phase,
        }

    def build_aux_cpc_head(self, config_rl):
        """The contrastive head, or None. Call AFTER `build_aux_canon_head`.

        Shares `aux_canon_site` and `aux_canon_parts` with the regression head,
        so the two objectives can be compared with one variable changed.
        """
        self.aux_cpc_head = None
        if not self.aux_cpc_enabled:
            return None
        if self.aux_canon_site in ("memory", "memory_obs", "probe"):
            self.head.expose_memory_embeds = True
        if self.aux_canon_site == "memory":
            in_size = self.head.memory_embed_size
        elif self.aux_canon_site in ("memory_obs", "probe"):
            in_size = self.head.encoded_obs_size + self.head.memory_embed_size
        else:
            in_size = self.head.embedding_size
        self.aux_cpc_head = AuxCanonCPC(
            embed_size=in_size,
            label_size=label_width(
                self.aux_canon_use_stone,
                self.aux_canon_use_potion,
                self.aux_canon_use_graph,
            ),
            proj_dim=int(getattr(config_rl, "aux_cpc_proj_dim", 128)),
            tau=float(getattr(config_rl, "aux_cpc_tau", 0.1)),
        )
        return self.aux_cpc_head

    def aux_cpc_loss(self, aux_embeds, targets, loss_mask):
        """InfoNCE between the memory read-out and the canonical label."""
        labels = encode_label(
            targets[:-1],
            self.aux_canon_use_stone,
            self.aux_canon_use_potion,
            self.aux_canon_use_graph,
        )
        return self.aux_cpc_head(aux_embeds[:-1], labels, loss_mask)

    def build_aux_canon_head(self, config_rl):
        """The auxiliary head, or None. Call AFTER `self.head` exists."""
        self.aux_canon_head = None
        if not self.aux_canon_enabled:
            return None
        if self.aux_canon_site in ("memory", "memory_obs", "probe"):
            if self.head.memory_embed_size <= 0:
                raise ValueError(
                    f"aux_canon_site={self.aux_canon_site!r} needs a sequence "
                    f"model with a memory readout; {self.head.seq_model.name!r} "
                    "has none"
                )
            self.head.expose_memory_embeds = True
        if self.aux_canon_site == "memory":
            in_size = self.head.memory_embed_size
        elif self.aux_canon_site in ("memory_obs", "probe"):
            in_size = self.head.encoded_obs_size + self.head.memory_embed_size
        else:
            in_size = self.head.embedding_size
        self.aux_canon_head = FlattenMlp(
            input_size=in_size,
            output_size=(
                AUX_CANON_STONE_OUT * int(self.aux_canon_use_stone)
                + AUX_CANON_POTION_OUT * int(self.aux_canon_use_potion)
                + AUX_CANON_GRAPH_OUT * int(self.aux_canon_use_graph)
            ),
            hidden_sizes=config_rl.config_critic.hidden_dims,
        )
        return self.aux_canon_head

    # ---- the leak guard ---------------------------------------------------
    def strip_aux_target(self, observs):
        """Excise the label block. EVERY path that hands an observation to the
        network must go through here, or the agent can read the answer key."""
        if not self.aux_canon_target:
            return observs
        assert observs.shape[-1] == self.obs_dim, (
            f"expected raw obs width {self.obs_dim}, got {observs.shape[-1]}"
        )
        return torch.cat(
            (observs[..., :self._aux_start], observs[..., self._aux_end:]),
            dim=-1,
        )

    def aux_target_slice(self, observs):
        assert observs.shape[-1] == self.obs_dim
        return observs[..., self._aux_start:self._aux_end]

    # ---- site dispatch ----------------------------------------------------
    def aux_canon_embeds(self, joint_embeds, memory_embeds, encoded_obs):
        if self.aux_canon_site == "joint":
            return joint_embeds
        if memory_embeds is None:
            raise RuntimeError(
                f"aux_canon_site={self.aux_canon_site!r} but RNN_head exposed "
                "no _memory_embeds"
            )
        if self.aux_canon_site == "memory":
            return memory_embeds
        if encoded_obs is None:
            raise RuntimeError(
                f"aux_canon_site={self.aux_canon_site!r} but RNN_head exposed "
                "no _encoded_obs"
            )
        memory_in = (
            memory_embeds.detach()
            if self.aux_canon_site == "probe"
            else memory_embeds
        )
        return torch.cat((encoded_obs.detach(), memory_in), dim=-1)

    # ---- metrics ----------------------------------------------------------
    def add_per_trial_metrics(self, metrics, name, hit, mask):
        n_trials = self._alchemy_num_trials
        length = self._alchemy_steps_per_trial
        if n_trials <= 1 or length <= 0 or hit.shape[0] != n_trials * length + 1:
            return
        num = (hit * mask)[1:].reshape(n_trials, length, *hit.shape[1:])
        den = mask[1:].reshape(n_trials, length, *mask.shape[1:])
        per_trial = num.flatten(1).sum(-1) / den.flatten(1).sum(-1).clamp(min=1.0)
        for i in range(n_trials):
            metrics[f"{name}_trial{i}"] = per_trial[i]

    # ---- loss -------------------------------------------------------------
    def aux_canon_loss(self, aux_embeds, observs, targets, loss_mask):
        """Masked MSE on stone coords, masked CE on potion types, masked BCE on
        the graph. `observs` must already be stripped."""
        embeds = aux_embeds[:-1]
        targets = targets[:-1]
        lead = embeds.shape[:-1]
        stone_present, potion_present = present_flags_from_observation(
            observs[:-1], **self._alchemy_split_kwargs
        )
        stone_mask = stone_present.to(embeds.dtype) * loss_mask
        potion_mask = potion_present.to(embeds.dtype) * loss_mask

        out = self.aux_canon_head(embeds)
        cursor = 0
        aux_loss = torch.zeros((), device=embeds.device, dtype=embeds.dtype)
        metrics = {}

        if self.aux_canon_use_stone:
            pred = out[..., cursor:cursor + AUX_CANON_STONE_OUT].reshape(*lead, -1, 3)
            cursor += AUX_CANON_STONE_OUT
            tgt = targets[..., :AUX_CANON_STONE_DIM].reshape(*lead, -1, 3)
            denom = stone_mask.sum().clamp(min=1.0)
            loss = (((pred - tgt) ** 2).mean(-1) * stone_mask).sum() / denom
            aux_loss = aux_loss + loss
            metrics["aux_canon_stone_loss"] = loss.detach()
            with torch.no_grad():
                hit = ((pred > 0) == (tgt > 0)).to(embeds.dtype).mean(-1)
                metrics["aux_canon_stone_acc"] = (hit * stone_mask).sum() / denom

        if self.aux_canon_use_potion:
            pred = out[..., cursor:cursor + AUX_CANON_POTION_OUT].reshape(
                *lead, AUX_CANON_POTION_DIM, AUX_CANON_NUM_POTION_TYPES
            )
            cursor += AUX_CANON_POTION_OUT
            tgt = targets[
                ..., AUX_CANON_STONE_DIM:AUX_CANON_STONE_DIM + AUX_CANON_POTION_DIM
            ].long().clamp(0, AUX_CANON_NUM_POTION_TYPES - 1)
            denom = potion_mask.sum().clamp(min=1.0)
            ce = F.cross_entropy(
                pred.reshape(-1, AUX_CANON_NUM_POTION_TYPES),
                tgt.reshape(-1), reduction="none",
            ).reshape(potion_mask.shape)
            loss = (ce * potion_mask).sum() / denom
            aux_loss = aux_loss + loss
            metrics["aux_canon_potion_loss"] = loss.detach()
            with torch.no_grad():
                hit = (pred.argmax(-1) == tgt).to(embeds.dtype)
                metrics["aux_canon_potion_acc"] = (hit * potion_mask).sum() / denom
                self.add_per_trial_metrics(
                    metrics, "aux_canon_potion_acc", hit, potion_mask
                )

        if self.aux_canon_use_graph:
            pred = out[..., cursor:cursor + AUX_CANON_GRAPH_OUT]
            cursor += AUX_CANON_GRAPH_OUT
            tgt = targets[..., -AUX_CANON_GRAPH_DIM:]
            # An edge is defined at every step, so only the rollout mask applies.
            denom = loss_mask.sum().clamp(min=1.0) * AUX_CANON_GRAPH_DIM
            bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
            loss = (bce * loss_mask).sum() / denom
            aux_loss = aux_loss + loss
            metrics["aux_canon_graph_loss"] = loss.detach()
            with torch.no_grad():
                hit = ((pred > 0) == (tgt > 0.5)).to(embeds.dtype)
                metrics["aux_canon_graph_acc"] = (hit * loss_mask).sum() / denom

        metrics["aux_canon_loss"] = aux_loss.detach()
        return aux_loss, metrics
