"""Structured Q heads.

The default critic is a flat ``FlattenMlp(embedding -> action_dim)``. For
Symbolic Alchemy that throws away a strong prior: the 40 actions are not 40
unrelated labels but ``NO_OP + stone(3) x target(13)``, where target 0 is the
cauldron and targets 1..12 are the potions. A flat head has to rediscover, in
39 independent output rows, that "potion 7 is currently useless" applies to all
three stones and that "stone 2 is already at the goal" applies to all thirteen
of its targets.
"""

import torch
import torch.nn as nn

from torchkit.networks import FlattenMlp


class FactoredAlchemyQHead(nn.Module):
    """Dueling + factored Q head over the ``NO_OP + stones x targets`` grid.

    One shared trunk emits five pieces, which are recombined as

        Q[no_op]   = V + A_no_op
        Q[i, j]    = V + A_stone[i] + A_target[j] + A_pair[i, j]

    with each advantage term mean-centered so ``V`` carries the state value and
    the advantages carry only the ranking (Wang+ 2016, arXiv:1511.06581).

    ``A_pair`` alone already spans every function a flat head can represent, so
    this is strictly no less expressive. The gain is the low-order terms: a fact
    about one stone is learned once and shared across its 13 targets instead of
    13 times, and likewise for a potion across the 3 stones. Alchemy is a case
    where most actions have near-identical value at any moment, which is exactly
    the regime dueling was introduced for.

    Output shape and semantics are identical to the flat head -- ``(..., 1 +
    max_stones * targets_per_stone)`` raw (pre-PopArt-affine) Q values -- so
    this is a drop-in replacement and nothing downstream changes.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes,
        *,
        max_stones: int,
        targets_per_stone: int,
    ):
        super().__init__()
        self.max_stones = int(max_stones)
        self.targets_per_stone = int(targets_per_stone)
        self.action_dim = 1 + self.max_stones * self.targets_per_stone

        self._n_v = 1
        self._n_noop = 1
        self._n_stone = self.max_stones
        self._n_target = self.targets_per_stone
        self._n_pair = self.max_stones * self.targets_per_stone

        self.trunk = FlattenMlp(
            input_size=input_size,
            output_size=(
                self._n_v
                + self._n_noop
                + self._n_stone
                + self._n_target
                + self._n_pair
            ),
            hidden_sizes=hidden_sizes,
        )

    def forward(self, *inputs) -> torch.Tensor:
        out = self.trunk(*inputs)
        v, a_noop, a_stone, a_target, a_pair = torch.split(
            out,
            [self._n_v, self._n_noop, self._n_stone, self._n_target, self._n_pair],
            dim=-1,
        )

        # Center each advantage term: identifiability, and it keeps V on the
        # value scale PopArt's statistics are tracking.
        a_stone = a_stone - a_stone.mean(dim=-1, keepdim=True)
        a_target = a_target - a_target.mean(dim=-1, keepdim=True)
        a_pair = a_pair - a_pair.mean(dim=-1, keepdim=True)

        a_pair = a_pair.unflatten(-1, (self.max_stones, self.targets_per_stone))
        grid = (
            a_pair
            + a_stone.unsqueeze(-1)      # broadcast over targets
            + a_target.unsqueeze(-2)     # broadcast over stones
        ).flatten(-2)

        advantage = torch.cat((a_noop, grid), dim=-1)
        advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        return v + advantage
