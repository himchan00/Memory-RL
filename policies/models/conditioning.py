"""
Conditioning modules for memory-augmented RL.

Three classes (ConcatConditioner / FiLMConditioner / HyperConditioner)
sharing one API: `forward(x: Tensor, c: Tensor | None) -> Tensor` and `.out_dim`.
RNN_head dispatches by name; stack depth = `len(hidden_sizes) + 1`.

References:
  - FiLM:       Perez et al., "FiLM: Visual Reasoning with a General
                Conditioning Layer" (AAAI 2018, arXiv:1709.07871).
  - HyperNet:   Ha et al., "HyperNetworks" (ICLR 2017, arXiv:1609.09106).
  - HFI init:   Chang et al., "Principled Weight Initialization for
                Hypernetworks" (ICLR 2020 / arXiv:2312.08399). §4.1 Case 2 is
                what we instantiate here: hypernet generates both W and b,
                so the variance budget is split (factor 1/2) between the
                weight head and the bias head. 
  - Meta-RL motivation:
                Beck et al., "Recurrent Hypernetworks are Surprisingly Strong
                in Meta-RL" (NeurIPS 2023, arXiv:2309.14970).
"""
import math
import torch
import torch.nn as nn


# ── Single-layer modulation primitives ─────────────────────────────────────


class FiLMLayer(nn.Module):
    """out = (1 + γ)·x + β,  (γ, β) = Linear(c). Zero-init → identity at start."""
    # Referenced from: Perez+ 2017 (arXiv:1709.07871).
    def __init__(self, cond_dim: int, x_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * x_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (..., x_dim), c: (..., cond_dim) — share leading dims (T, B) or (B,)
        gamma, beta = self.proj(c).chunk(2, dim=-1)
        return (1.0 + gamma) * x + beta


class HyperLinear(nn.Module):
    """A linear layer whose (W, b) are generated per sample from condition c."""
    # Refs:
    #   Ha+ 2016 (arXiv:1609.09106): hypernet concept.
    #   Chang+ 2020 (arXiv:2312.08399) Table 1: Hyperfan-In (HFI) for Case 2
    #   (hypernet generates both W and b) with ReLU base activation gives
    #     Var(W_head) = 1/(d_cond·d_in),  Var(b_head) = 1/d_cond,
    #   ⇒ uniform U(-a, a) with a = sqrt(3·Var).
    def __init__(self, cond_dim: int, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.hyper_w = nn.Linear(cond_dim, in_dim * out_dim)
        self.hyper_b = nn.Linear(cond_dim, out_dim)
        bound_w = math.sqrt(3.0 / (cond_dim * in_dim))
        bound_b = math.sqrt(3.0 / (cond_dim))
        nn.init.uniform_(self.hyper_w.weight, -bound_w, bound_w)
        nn.init.uniform_(self.hyper_b.weight, -bound_b, bound_b)
        # β, γ (head biases) zero per paper.
        nn.init.zeros_(self.hyper_w.bias)
        nn.init.zeros_(self.hyper_b.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        W = self.hyper_w(c).reshape(*c.shape[:-1], self.out_dim, self.in_dim)
        b = self.hyper_b(c)
        return torch.einsum("...oi,...i->...o", W, x) + b


class ConcatConditioner(nn.Module):
    """n blocks of (Linear → activation), then `cat(., c)`.

    `out_dim = mlp_out_dim + cond_dim`; `cond_dim=0` (markov) reduces to MLP.
    """
    def __init__(self, in_dim, out_dim, hidden_sizes, cond_dim,
                 activation=nn.LeakyReLU):
        super().__init__()
        dims_in = [in_dim] + list(hidden_sizes)
        dims_out = list(hidden_sizes) + [out_dim]
        self.lins = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(dims_in, dims_out)
        )
        self.act = activation()
        self.out_dim = out_dim + cond_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        for lin in self.lins:
            x = self.act(lin(x))
        if c is None:
            return x
        return torch.cat([x, c], dim=-1)


class FiLMConditioner(nn.Module):
    """n blocks of (Linear → activation → FiLM(·, c)).

    n = len(hidden_sizes) + 1.  Default hidden_sizes=() → single block.
    """
    def __init__(self, in_dim, out_dim, hidden_sizes, cond_dim,
                 activation=nn.LeakyReLU):
        super().__init__()
        dims_in = [in_dim] + list(hidden_sizes)
        dims_out = list(hidden_sizes) + [out_dim]
        self.lins = nn.ModuleList(
            nn.Linear(i, o) for i, o in zip(dims_in, dims_out)
        )
        self.films = nn.ModuleList(
            FiLMLayer(cond_dim, o) for o in dims_out
        )
        self.act = activation()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for lin, film in zip(self.lins, self.films):
            x = film(self.act(lin(x)), c)
        return x


class HyperConditioner(nn.Module):
    """n blocks of (HyperLinear(·, c) → activation), activation applied every block.

    n = len(hidden_sizes) + 1.  Default hidden_sizes=() → single HyperLinear.

    Referenced from Beck+ 2023 (jacooba/hyper, models/policy.py L442-446): the
    hypernet stack applies activation after every Linear, including the last.
    Symmetric with FiLMConditioningStack which also activates every block.
    """
    def __init__(self, in_dim, out_dim, hidden_sizes, cond_dim,
                 activation=nn.LeakyReLU):
        super().__init__()
        dims_in = [in_dim] + list(hidden_sizes)
        dims_out = list(hidden_sizes) + [out_dim]
        self.hypers = nn.ModuleList(
            HyperLinear(cond_dim, i, o) for i, o in zip(dims_in, dims_out)
        )
        self.act = activation()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for hyper in self.hypers:
            x = self.act(hyper(x, c))
        return x
