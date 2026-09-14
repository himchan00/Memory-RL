"""Supervised contrastive aux: align the memory with the canonical chemistry.

The repo already has both halves of this idea, separately. `msc_v2_aux.py` runs
InfoNCE with NO labels: two disjoint random halves of one episode must recognise
each other against other episodes, so whatever they share -- the episode's hidden
chemistry -- is what survives in the memory. `aux_canon.py` uses the labels, but
asks the memory to REGRESS them.

This is the missing corner: labels AND InfoNCE.

    f(m_t) . g(y_t)^T / kappa ,  positives on the diagonal of the batch

`aux_canon` makes the memory reproduce the label's exact parameterisation --
these nine floats, those twelve class indices. InfoNCE only asks the memory to
be DISCRIMINATIVE of the chemistry: any encoding that tells this episode's
chemistry apart from the other 31 in the batch scores perfectly, whatever
coordinate system it is written in. That is a strictly weaker demand on a memory
that already holds the information, which is the case measured for MATE by
`scripts/train_world_model.py`'s probe.

The label is the same 33-dim block `aux_canon` uses, so `aux_canon_parts` and
`aux_canon_site` select the same slices and the same read-out point, and the two
losses differ in exactly one thing: the objective.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.alchemy import (
    AUX_CANON_ABSENT,
    AUX_CANON_GRAPH_DIM,
    AUX_CANON_NUM_POTION_TYPES,
    AUX_CANON_POTION_DIM,
    AUX_CANON_STONE_DIM,
)


def encode_label(targets, use_stone, use_potion, use_graph):
    """(..., 33) raw label block -> (..., D) a network can consume.

    Potion slots hold a class index, not a magnitude, so they are one-hot with an
    extra ABSENT column; stone coordinates and graph edges are already numeric.
    Feeding the raw -99.0 sentinel to an MLP would let one absent slot dominate
    every downstream activation.
    """
    parts = []
    if use_stone:
        stone = targets[..., :AUX_CANON_STONE_DIM]
        absent = stone <= AUX_CANON_ABSENT / 2.0
        parts.append(torch.where(absent, torch.zeros_like(stone), stone))
        parts.append(absent.to(targets.dtype))
    if use_potion:
        potion = targets[
            ..., AUX_CANON_STONE_DIM:AUX_CANON_STONE_DIM + AUX_CANON_POTION_DIM
        ]
        absent = potion <= AUX_CANON_ABSENT / 2.0
        idx = torch.where(
            absent,
            torch.full_like(potion, AUX_CANON_NUM_POTION_TYPES),
            potion.clamp(0, AUX_CANON_NUM_POTION_TYPES - 1),
        ).long()
        parts.append(
            F.one_hot(idx, AUX_CANON_NUM_POTION_TYPES + 1)
            .reshape(*targets.shape[:-1], -1)
            .to(targets.dtype)
        )
    if use_graph:
        parts.append(targets[..., -AUX_CANON_GRAPH_DIM:])
    return torch.cat(parts, dim=-1)


def label_width(use_stone, use_potion, use_graph):
    return (
        (2 * AUX_CANON_STONE_DIM) * int(use_stone)
        + (AUX_CANON_POTION_DIM * (AUX_CANON_NUM_POTION_TYPES + 1)) * int(use_potion)
        + AUX_CANON_GRAPH_DIM * int(use_graph)
    )


class AuxCanonCPC(nn.Module):
    """InfoNCE between a memory read-out and the canonical chemistry label."""

    def __init__(self, embed_size, label_size, proj_dim=128, tau=0.1):
        super().__init__()
        if not math.isfinite(float(tau)) or tau <= 0.0:
            raise ValueError("aux_cpc_tau must be finite and positive")
        self.f = nn.Sequential(
            nn.Linear(embed_size, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.g = nn.Sequential(
            nn.Linear(label_size, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )
        self.log_kappa = nn.Parameter(torch.tensor(math.log(float(tau))))

    def forward(self, embeds, labels, loss_mask):
        """embeds (T,B,E), labels (T,B,L), loss_mask (T,B,1) -> loss, metrics.

        The negatives at step t are the OTHER EPISODES at that same step, so the
        loss cannot be won by reading the timestep off the memory -- every
        candidate shares it. Only the chemistry separates them.
        """
        batch_size = embeds.shape[1]
        if batch_size < 2:
            raise ValueError("aux CPC needs at least two episodes per batch")

        # A step is usable only if EVERY episode is valid there; otherwise the
        # padded rows would enter as negatives and the diagonal would not be the
        # only positive. Alchemy episodes are fixed length, so this keeps them all.
        step_ok = (loss_mask.squeeze(-1) > 0).all(dim=1).to(embeds.dtype)  # (T,)
        denom = step_ok.sum().clamp(min=1.0)

        q = F.normalize(self.f(embeds), dim=-1)                    # (T,B,P)
        k = F.normalize(self.g(labels), dim=-1)                    # (T,B,P)
        kappa = self.log_kappa.exp().clamp_min(1e-6)
        logits = torch.bmm(q, k.transpose(1, 2)) / kappa           # (T,B,B)

        tgt = torch.arange(batch_size, device=embeds.device)
        tgt = tgt.unsqueeze(0).expand(logits.shape[0], -1)         # (T,B)
        ce_f = F.cross_entropy(
            logits.reshape(-1, batch_size), tgt.reshape(-1), reduction="none"
        ).reshape(tgt.shape).mean(dim=1)
        ce_b = F.cross_entropy(
            logits.transpose(1, 2).reshape(-1, batch_size), tgt.reshape(-1),
            reduction="none",
        ).reshape(tgt.shape).mean(dim=1)
        loss = 0.5 * ((ce_f + ce_b) * step_ok).sum() / denom

        with torch.no_grad():
            hit = (logits.argmax(dim=2) == tgt).to(embeds.dtype).mean(dim=1)
            acc = (hit * step_ok).sum() / denom
        return loss, {
            "aux_cpc_loss": loss.detach(),
            "aux_cpc_acc": acc,
            "aux_cpc_chance": torch.full_like(acc, 1.0 / batch_size),
            # InfoNCE is a lower bound on I(memory; chemistry) in nats.
            "aux_cpc_mi_lower_bound": loss.new_tensor(math.log(batch_size))
            - loss.detach(),
            "aux_cpc_kappa": kappa.detach(),
        }
