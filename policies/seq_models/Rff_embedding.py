"""
Random Fourier Feature (RFF) embedding layer for MATE.
 
Implements the cos&sin estimator (Sutherland & Schneider, UAI 2015), which
has strictly lower variance than the random-phase variant for the kernel-value
range that matters in practice.
 
Supported kernels (all shift-invariant):
  - 'gaussian': K(x, y) = exp(-||x - y||^2 / (2 sigma^2))
                spectral measure: N(0, sigma^{-2} I)
  - 'laplace':  K(x, y) = exp(-||x - y||_1 / sigma)
                spectral measure: independent Cauchy(0, 1/sigma) per coordinate
  - 'matern':   Matern with smoothness nu in {0.5, 1.5, 2.5, ...}
                spectral measure: multivariate Student-t
                (nu=0.5 reduces to Laplace with l2 distance)
  - 'riesz':    K(x, y) = -||x - y||  (conditionally positive definite)
                spectral measure: regularized power-law |omega|^{-(n+1)}.
                We sample radius from a regularized inverse-power density and
                direction uniformly on the sphere.  This gives a *proxy* for
                the Riesz kernel that retains the (n+1)-Hoelder reverse W_1
                bound up to a regularization-dependent constant.
"""
 
from __future__ import annotations
 
import math
from typing import Literal
 
import torch
import torch.nn as nn
 
KernelName = Literal["gaussian", "laplace", "matern", "riesz"]
 
 
def _sample_gaussian_frequencies(num_freq: int, input_dim: int, sigma: float,
                                 generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Spectral measure of the Gaussian kernel exp(-||x-y||^2 / (2 sigma^2))
    is N(0, sigma^{-2} I).  Standard fact (e.g. Rahimi & Recht 2007 Sec. 2.1).
    """
    omega = torch.randn(num_freq, input_dim, generator=generator)
    return omega / sigma
 
 
def _sample_cauchy_frequencies(num_freq: int, input_dim: int, sigma: float,
                               generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Spectral measure of the (l1-)Laplace kernel exp(-||x-y||_1 / sigma) is
    a product of standard Cauchy distributions, scaled by 1/sigma.
 
    Cauchy(0, gamma) is sampled via the inverse CDF transform:
        u ~ Uniform(0, 1) -> X = gamma * tan(pi (u - 1/2)).
    """
    u = torch.rand(num_freq, input_dim, generator=generator)
    cauchy = torch.tan(math.pi * (u - 0.5))
    return cauchy / sigma
 
 
def _sample_student_t_frequencies(num_freq: int, input_dim: int, df: float, sigma: float,
                                  generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Spectral measure of the Matern-nu kernel is a multivariate Student-t with
    df = 2 nu degrees of freedom, scaled by 1/sigma.
 
    Student-t with df = nu can be sampled as:
        Z ~ N(0, I), V ~ Chi2(df) -> X = Z * sqrt(df / V).
    Chi2(df) is sampled via Gamma(df/2, 2).
    """
    z = torch.randn(num_freq, input_dim, generator=generator)
    gamma_dist = torch.distributions.Gamma(concentration=df / 2.0, rate=0.5)
    # Gamma.sample does not accept a generator argument in current PyTorch,
    # so we draw from torch.empty().exponential_-style routines via the dist.
    v = gamma_dist.sample((num_freq,))
    t_sample = z * torch.sqrt(df / v).unsqueeze(-1)
    return t_sample / sigma
 
 
def _sample_riesz_frequencies(num_freq: int, input_dim: int, eps: float,
                              generator: torch.Generator | None = None) -> torch.Tensor:
    """
    The Riesz / energy kernel K(x, y) = -||x - y|| has spectral density
    proportional to ||omega||^{-(n+1)} on R^n, which is not integrable.
    We use a regularized proxy: sample direction uniformly on S^{n-1}, and
    radius from the density
 
        p(r) propto r^{-(n+1)} on r >= eps,    p(r) = 0 on r < eps.
 
    This is a Pareto distribution with shape alpha = n and scale eps:
        r = eps * U^{-1/n},   U ~ Uniform(0, 1).
 
    The cutoff eps regularizes the singularity at the origin; smaller eps
    -> closer to the true Riesz spectral measure but larger frequencies and
    therefore higher gradient variance.  eps in [1e-2, 1.0] works well in
    practice.
    """
    direction = torch.randn(num_freq, input_dim, generator=generator)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
 
    u = torch.rand(num_freq, generator=generator)
    radius = eps * u.pow(-1.0 / input_dim)  # Pareto(alpha=n, scale=eps)
 
    return direction * radius.unsqueeze(-1)
 
 
class RFFEmbedding(nn.Module):
    """
    Random Fourier Feature embedding using the cos&sin estimator.
 
    Given input x in R^{input_dim}, returns z(x) in R^{embedding_dim}
    of the form
 
        z(x) = sqrt(1/D) * [cos(w_1 x), sin(w_1 x), ..., cos(w_D x), sin(w_D x)]
 
    with D = embedding_dim // 2 frequencies w_k drawn i.i.d. from the spectral
    measure of the chosen kernel.  The frequencies are stored as a frozen
    buffer; only the input encoder upstream of this layer is trained.
 
    The expected inner product satisfies
        E[z(x)^T z(y)] = K(x, y)
    where K is the selected kernel (or its regularized proxy for 'riesz').
    Hence the L2 distance between mean-pooled RFF embeddings of two
    multisets approximates the MMD with kernel K (Sriperumbudur et al.
    JMLR 2010; Sutherland & Schneider UAI 2015).
 
    Args:
        input_dim:     dimension n of the input space.
        embedding_dim: output dimension of z(x).  Must be even.
        kernel:        one of 'gaussian', 'laplace', 'matern', 'riesz'.
        sigma:         bandwidth / length-scale (ignored for 'riesz').
                       Default 1.0; the user is responsible for scaling
                       inputs appropriately.
        matern_nu:     smoothness parameter for the Matern kernel.
                       Common choices: 0.5 (Laplace), 1.5, 2.5.  Default 1.5.
        riesz_eps:     low-frequency cutoff for the Riesz proxy.  Default 0.1.
        seed:          optional integer seed for reproducible frequency draws.
 
    Forward input shape:  (..., input_dim)
    Forward output shape: (..., embedding_dim)
    """
 
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        kernel: KernelName = "gaussian",
        sigma: float = 1.0,
        matern_nu: float = 1.5,
        riesz_eps: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__()
 
        if embedding_dim % 2 != 0:
            raise ValueError(
                f"embedding_dim must be even (got {embedding_dim}); "
                "the cos&sin estimator emits 2 features per frequency."
            )
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1 (got {input_dim}).")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0 (got {sigma}).")
 
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_freq = embedding_dim // 2
        self.kernel = kernel
        self.sigma = sigma
        self.matern_nu = matern_nu
        self.riesz_eps = riesz_eps
 
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)

        print(f"Initializing RFF embedding with kernel='{kernel}'")
        if kernel == "gaussian":
            omega = _sample_gaussian_frequencies(self.num_freq, input_dim, sigma, gen)
        elif kernel == "laplace":
            omega = _sample_cauchy_frequencies(self.num_freq, input_dim, sigma, gen)
        elif kernel == "matern":
            omega = _sample_student_t_frequencies(
                self.num_freq, input_dim, df=2.0 * matern_nu, sigma=sigma, generator=gen
            )
        elif kernel == "riesz":
            omega = _sample_riesz_frequencies(self.num_freq, input_dim, riesz_eps, gen)
        else:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Expected one of: "
                "gaussian, laplace, matern, riesz."
            )
 
        # Frozen frequencies; not trainable.
        self.register_buffer("omega", omega)
        # Precompute normalization constant.
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(self.num_freq))
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}."
            )
        # Linear projection: (..., input_dim) @ (input_dim, num_freq) -> (..., num_freq).
        proj = x @ self.omega.T
        cos_part = torch.cos(proj)
        sin_part = torch.sin(proj)
        # Interleave so that adjacent pairs correspond to the same frequency.
        # Output: (..., 2 * num_freq) = (..., embedding_dim).
        out = torch.stack([cos_part, sin_part], dim=-1).flatten(start_dim=-2)
        return self.scale * out
 
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, embedding_dim={self.embedding_dim}, "
            f"num_freq={self.num_freq}, kernel='{self.kernel}', sigma={self.sigma}"
        )