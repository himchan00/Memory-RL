"""
Random Fourier Feature (RFF) embedding layer for MATE.
 
Implements the cos&sin estimator (Sutherland & Schneider, UAI 2015), which
has strictly lower variance than the random-phase variant for the kernel-value
range that matters in practice.
 
Supported kernels (all shift-invariant):
  - 'gaussian': K(x, y) = exp(-||x - y||_2^2 / (2 sigma^2))
                Positive definite.  Spectral measure: N(0, sigma^{-2} I).
  - 'laplace':  K(x, y) = exp(-||x - y||_1 / sigma)
                Positive definite, separable.  Spectral measure: independent
                Cauchy(0, 1/sigma) per coordinate.
                Note: this is the l1-Laplace kernel.  It is *not* the same as
                the l2-Laplace kernel exp(-||x-y||_2 / sigma), which would be
                obtained from Matern with nu=0.5.
  - 'matern':   Matern with smoothness nu in {0.5, 1.5, 2.5, ...}.
                Positive definite, isotropic in l2.  Spectral measure:
                multivariate Student-t.  nu=0.5 reduces to the l2-Laplace
                kernel exp(-||x-y||_2 / sigma).
  - 'train':    Learnable frequencies.  Initialized from the Gaussian spectral
                measure N(0, sigma^{-2} I) (same as kernel='gaussian'), but
                ``omega`` is registered as an nn.Parameter and learned end-to-end
                via downstream gradients.  Importance weights sqrt(w_k) are
                fixed to 1.  No kernel-MMD interpretation: this turns the RFF
                layer into a (cos, sin)-activated random projection whose
                projection matrix is trained jointly with the rest of the model.

 
Expected estimator properties:
    E[z(x)^T z(y)] = K(x, y)              (unbiased kernel estimator)
so the L2 distance between mean-pooled embeddings of two multisets
approximates the MMD (Sriperumbudur et al. JMLR 2010; Sutherland &
Schneider UAI 2015).
"""
 
from __future__ import annotations
 
import math
from typing import Literal
 
import torch
import torch.nn as nn

KernelName = Literal["gaussian", "laplace", "matern", "train"]
 
 
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
    # so reproducibility for matern requires seeding the global RNG.
    v = gamma_dist.sample((num_freq,))
    t_sample = z * torch.sqrt(df / v).unsqueeze(-1)
    return t_sample / sigma
 
 
 
class RFFEmbedding(nn.Module):
    """
    Random Fourier Feature embedding using the cos&sin estimator.
 
    Given input x in R^{input_dim}, returns z(x) in R^{embedding_dim}
    of the form

        z(x) = sqrt(2.0) * [sqrt(w_1) cos(w_1 x), sqrt(w_1) sin(w_1 x),
                ...,
                sqrt(w_D) cos(w_D x), sqrt(w_D) sin(w_D x)]

    where D = embedding_dim // 2.  Frequencies w_k are drawn i.i.d. from
    the spectral measure of the chosen kernel.

    Note: the canonical MMD-unbiased estimator carries an overall
    1/ sqrt(2D) prefactor so that E[z(x)^T z(y)] = K(x, y) exactly.  We
    drop that prefactor here to keep activations in the natural [-1, 1]
    range, which empirically interacts better with downstream LayerNorms
    / value heads.  The pairwise identity then holds up to a constant
    factor 2D; the *direction* of memory differences (which is what MATE
    consumes) is unchanged.

    Estimator semantics (up to the constant 2D from dropping sqrt(1/2D)):
        E[z(x)^T z(y)] propto K(x, y)
    so for any pair of multisets mu, nu,
        E[ ||z_bar(mu) - z_bar(nu)||^2 ] propto MMD_K^2(mu, nu).
 
    Args:
        input_dim:        dimension n of the input space.
        embedding_dim:    output dimension of z(x).  Must be even.
        kernel:           one of 'gaussian', 'laplace', 'matern', 'train'.
        sigma:            bandwidth / length-scale.
                          Default sqrt(input_dim); which typically matches the order of magnitude of ||x-y||
        matern_nu:        smoothness parameter for the Matern kernel.
                          Common choices: 0.5 (l2-Laplace), 1.5, 2.5. (infinite nu corresponds to gaissian kernel)
                          Default 1.5.
        seed:             optional integer seed for reproducible frequency
                          draws.

    Forward input shape:  (..., input_dim)
    Forward output shape: (..., embedding_dim)
    """
 
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        kernel: KernelName = "gaussian",
        sigma: float | None = None,
        matern_nu: float = 1.5,
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
        if sigma is None:
            sigma = math.sqrt(input_dim)  # default heuristic: match typical ||x-y|| scale
        elif sigma <= 0:
            raise ValueError(f"sigma must be > 0 (got {sigma}).")
 
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_freq = embedding_dim // 2
        self.kernel = kernel
        self.sigma = sigma
        self.matern_nu = matern_nu
 
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
 
 
        print(f"Initializing RFF embedding with kernel='{kernel}'")
        if kernel == "gaussian":
            omega = _sample_gaussian_frequencies(
                self.num_freq, input_dim, sigma, generator=gen
            )
        elif kernel == "laplace":
            omega = _sample_cauchy_frequencies(
                self.num_freq, input_dim, sigma, generator=gen
            )
        elif kernel == "matern":
            omega = _sample_student_t_frequencies(
                self.num_freq, input_dim, df=2.0 * matern_nu, sigma=sigma,
                generator=gen,
            )
        elif kernel == "train":
            # Learnable frequencies; initialized from the Gaussian spectral
            # measure.  Registered as nn.Parameter below.
            omega = _sample_gaussian_frequencies(
                self.num_freq, input_dim, sigma, generator=gen
            )
        else:
            raise ValueError(
                f"Unknown kernel '{kernel}'. Expected one of: "
                "gaussian, laplace, matern, train."
            )
 
        # Frequencies: learnable when kernel='train', frozen buffer otherwise.
        if kernel == "train":
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}."
            )
        # Linear projection: (..., input_dim) @ (input_dim, num_freq) -> (..., num_freq).
        proj = x @ self.omega.T
        out = math.sqrt(2.0) * torch.stack([torch.cos(proj), torch.sin(proj)], dim=-1).flatten(start_dim=-2)
        return out