"""Slot-shared (DeepSets-style) observation encoder for Symbolic Alchemy.

The symbolic observation is a flat vector that is really a set of slots: 3 stone
slots of ``[c0, c1, c2, reward/3, used]`` followed by 12 potion slots. Read
positionally -- which is what the default flat MLP does -- the network has to
learn "what a potion is" twelve separate times, once per input offset, and any
fact it learns about the potion in slot 3 transfers to slot 7 only by accident.
Alchemy shuffles which slot holds which potion every trial, so that accident
never becomes reliable.

This applies a single shared per-slot MLP to every stone slot and another to
every potion slot, so "how to read a slot" is learned once from 3x (resp. 12x)
the data. Zaheer+ 2017 (arXiv:1703.06114).

Crucially it does NOT pool the per-slot embeddings away: the action head is
positional (action ``1 + 13*i + j`` names stone slot ``i`` and potion slot
``j``), so slot identity has to survive. The per-slot embeddings are emitted in
slot order and a mean-pooled summary of each group is appended alongside, giving
a permutation-EQUIVARIANT encoding rather than an invariant one.

NON-PRIVILEGED: this is a re-parameterization of the agent's own observation.
The trailing bytes -- trial flag, trial phase, and the oracle context tail when
one is present -- bypass the encoder untouched.
"""

import torch
import torch.nn as nn

from envs.alchemy import TRIAL_PHASE_DIM, get_symbolic_alchemy_layout


def _slot_mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LeakyReLU(),
        nn.Linear(hidden, out_dim),
        nn.LeakyReLU(),
    )


class AlchemySlotEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        observe_used: bool,
        add_trial_flag: bool,
        add_trial_phase: bool,
        structured_potions: bool,
        context_dim: int = 0,
        slot_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        layout = get_symbolic_alchemy_layout(observe_used, structured_potions)
        self.max_stones = layout.max_stones
        self.max_potions = layout.max_potions
        self.stone_feature_dim = layout.stone_feature_dim
        self.potion_feature_dim = layout.potion_feature_dim
        self.stone_width = self.max_stones * self.stone_feature_dim
        self.potion_width = self.max_potions * self.potion_feature_dim

        self.tail_dim = (
            int(obs_dim)
            - self.stone_width
            - self.potion_width
        )
        expected_tail = (
            int(add_trial_flag)
            + (TRIAL_PHASE_DIM if add_trial_phase else 0)
            + int(context_dim)
        )
        if self.tail_dim != expected_tail or self.tail_dim < 0:
            raise ValueError(
                f"AlchemySlotEncoder cannot parse obs_dim={obs_dim}: stones "
                f"{self.stone_width} + potions {self.potion_width} leaves a "
                f"tail of {self.tail_dim}, but the flags imply {expected_tail}"
            )

        self.stone_mlp = _slot_mlp(self.stone_feature_dim, slot_dim, hidden_dim)
        self.potion_mlp = _slot_mlp(self.potion_feature_dim, slot_dim, hidden_dim)

        # per-slot embeddings (order preserved) + one pooled summary per group
        self.out_dim = (
            (self.max_stones + 1) * slot_dim
            + (self.max_potions + 1) * slot_dim
            + self.tail_dim
        )

    def forward(self, observs: torch.Tensor) -> torch.Tensor:
        lead = observs.shape[:-1]
        stones = observs[..., : self.stone_width].unflatten(
            -1, (self.max_stones, self.stone_feature_dim)
        )
        potions = observs[
            ..., self.stone_width : self.stone_width + self.potion_width
        ].unflatten(-1, (self.max_potions, self.potion_feature_dim))
        tail = observs[..., self.stone_width + self.potion_width :]

        s = self.stone_mlp(stones)    # (..., max_stones, slot_dim)
        p = self.potion_mlp(potions)  # (..., max_potions, slot_dim)

        return torch.cat(
            (
                s.flatten(-2),
                s.mean(dim=-2),
                p.flatten(-2),
                p.mean(dim=-2),
                tail,
            ),
            dim=-1,
        ).reshape(*lead, self.out_dim)
