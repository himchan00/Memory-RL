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
  - 'riesz':    K(x, y) = -||x - y||_2
                *Conditionally* positive definite (CPD).  Its MMD equals the
                energy distance and admits a sliced spectral decomposition
                (Hertrich et al., ICLR 2024, Theorem 2):
                    MMD_K^2(mu, nu) = c_n * E_v[ MMD_{K_1D}^2(P_v mu, P_v nu) ]
                We exploit this sliced structure to design an importance-sampling
                proposal: random direction v ~ Unif(S^{n-1}), 1D frequency
                xi ~ Cauchy(0, sigma_xi), with a regularization eps to keep
                importance weights 1/(xi^2 + eps^2) bounded.  Note: Hertrich
                et al. themselves use a *sort-based* 1D MMD computation, not
                RFF.  Our RFF estimator targets a regularized version of the
                same MMD, and the reverse W_1 Hoelder bound from Hertrich et
                al. applies in the eps -> 0 limit with constants depending on
                the regularization.
 
Expected estimator properties:
  - For PD kernels (gaussian, laplace, matern):
        E[z(x)^T z(y)] = K(x, y)              (unbiased kernel estimator)
    so the L2 distance between mean-pooled embeddings of two multisets
    approximates the MMD (Sriperumbudur et al. JMLR 2010; Sutherland &
    Schneider UAI 2015).
  - For the CPD Riesz kernel, the pairwise identity above does NOT hold
    (K is negative-valued, while sqrt(w)*cos*cos terms do not match it
    pointwise).  What does hold, by construction of the importance-weighted
    features, is
        E[ ||z_bar(mu) - z_bar(nu)||^2 ] = MMD_K^2(mu, nu)
    i.e. the squared L2 distance between mean-pooled embeddings is an
    unbiased estimator of the squared MMD (Hertrich et al. 2024, applied
    via standard importance-sampling Monte Carlo).
"""
 
from __future__ import annotations
 
import math
from typing import Literal
 
import torch
import torch.nn as nn
 
from torchkit.networks import InputNorm
 
KernelName = Literal["gaussian", "laplace", "matern", "riesz", "train"]
 
 
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
 
 
def _sample_sliced_riesz(
    num_freq: int,
    input_dim: int,
    sigma_xi: float = 1.0,
    eps: float = 0.1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sliced importance-sampled proposal for the Riesz / energy kernel.
 
        omega = xi * v,   v ~ Unif(S^{n-1}),   xi ~ Cauchy(0, sigma_xi).
 
    Hertrich et al. (ICLR 2024) show that the Riesz MMD admits a sliced
    spectral form whose 1D weight is 1/xi^2.  We regularize this weight as
    1/(xi^2 + eps^2) to keep importance weights bounded (otherwise a single
    xi sampled close to 0 would dominate the entire memory).
 
    Importance weight derivation (proposal q = Cauchy(0, sigma_xi) on xi,
    direction v uniform on the sphere):
 
        target(xi)   propto 1 / (xi^2 + eps^2)              [regularized Riesz]
        proposal(xi) =     1 / [pi * sigma_xi * (1 + (xi/sigma_xi)^2)]
        w(xi)        =  target(xi) / proposal(xi)
 
    The dimension-dependent absolute constant c_n is absorbed into the
    overall feature scale and does not affect downstream learning (the
    policy / value network learns to consume the relative magnitudes).
 
    Args:
        num_freq:  number of frequency samples D.
        input_dim: dimension n of the input space.
        sigma_xi:  scale of the 1D Cauchy proposal on xi.  Should roughly
                   match the typical scale of projections v^T x.
        eps:       regularization of the target weight; controls the maximum
                   weight magnitude.  Smaller eps -> closer to true Riesz
                   but higher variance.
        generator: optional torch.Generator for reproducibility.
 
    Returns:
        omegas: (num_freq, input_dim) tensor of frequencies.
        sqrt_w: (num_freq,) tensor of sqrt(importance weights).
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0 (got {eps}).")
    if sigma_xi <= 0:
        raise ValueError(f"sigma_xi must be > 0 (got {sigma_xi}).")
 
    # uniform direction on the unit sphere
    u = torch.randn(num_freq, input_dim, generator=generator)
    v = u / u.norm(dim=-1, keepdim=True)                       # (D, n)
 
    # 1D Cauchy frequency, via inverse-CDF
    U = torch.rand(num_freq, generator=generator)
    xi = sigma_xi * torch.tan(math.pi * (U - 0.5))             # (D,)
 
    omegas = xi[:, None] * v                                   # (D, n)
 
    # log importance weight: log target - log proposal
    log_target = -torch.log(xi.pow(2) + eps ** 2)
    log_proposal = -math.log(math.pi * sigma_xi) - torch.log1p((xi / sigma_xi) ** 2)
    log_w = log_target - log_proposal                          # (D,)
    sqrt_w = (0.5 * log_w).exp()                               # (D,)
 
    return omegas, sqrt_w
 
 
class RFFEmbedding(nn.Module):
    """
    Random Fourier Feature embedding using the cos&sin estimator.
 
    Given input x in R^{input_dim}, returns z(x) in R^{embedding_dim}
    of the form

        z(x) = sqrt(2.0) * [sqrt(w_1) cos(w_1 x), sqrt(w_1) sin(w_1 x),
                ...,
                sqrt(w_D) cos(w_D x), sqrt(w_D) sin(w_D x)]

    where D = embedding_dim // 2.  Frequencies w_k are drawn i.i.d. from
    the spectral measure of the chosen kernel (for PD kernels) or from a
    regularized importance-sampling proposal (for the CPD Riesz kernel).
    Importance weights sqrt(w_k) are stored as a frozen buffer; they are 1
    for the PD kernels and non-trivial only for Riesz.  Only the input
    encoder upstream of this layer is trained.

    Note: the canonical MMD-unbiased estimator carries an overall
    1/ sqrt(2D) prefactor so that E[z(x)^T z(y)] = K(x, y) exactly.  We
    drop that prefactor here to keep activations in the natural [-1, 1]
    range, which empirically interacts better with downstream LayerNorms
    / value heads.  The pairwise identity then holds up to a constant
    factor 2D; the *direction* of memory differences (which is what MATE
    consumes) is unchanged.

    Estimator semantics (up to the constant 2D from dropping sqrt(1/2D)):
      - PD kernels (gaussian, laplace, matern):
            E[z(x)^T z(y)] propto K(x, y)
        so for any pair of multisets mu, nu,
            E[ ||z_bar(mu) - z_bar(nu)||^2 ] propto MMD_K^2(mu, nu).
      - CPD kernel (riesz): the pairwise identity does not hold, but the
        multiset-distance identity does (up to the same constant):
            E[ ||z_bar(mu) - z_bar(nu)||^2 ] propto MMD_K^2(mu, nu)
        via importance sampling of the sliced spectral form (Hertrich et al.
        ICLR 2024).  This is the property MATE actually consumes.
 
    Args:
        input_dim:        dimension n of the input space.
        embedding_dim:    output dimension of z(x).  Must be even.
        kernel:           one of 'gaussian', 'laplace', 'matern', 'riesz', 'train'.
        sigma:            bandwidth / length-scale.  For 'riesz', this is
                          interpreted as the 1D Cauchy proposal scale
                          sigma_xi (the typical scale of v^T x).
                          Default 1.0; the user is responsible for scaling
                          inputs appropriately.
        matern_nu:        smoothness parameter for the Matern kernel.
                          Common choices: 0.5 (l2-Laplace), 1.5, 2.5.
                          Default 1.5.
        riesz_eps:        regularization of the target Riesz weight,
                          1 / (xi^2 + riesz_eps^2).  Controls the maximum
                          importance-weight magnitude.  Smaller -> closer
                          to true Riesz but higher variance.  Default 0.1.
                          Only used when kernel='riesz'.
        seed:             optional integer seed for reproducible frequency
                          draws.
        normalize_inputs: if True, apply a moving-average InputNorm to x
                          before the RFF projection.  Updates running stats
                          during training; supports an optional ``mask`` in
                          ``forward`` to exclude padded entries from the
                          statistics.  Note: since RFF frequencies are
                          frozen, early-training drift in InputNorm
                          statistics will shift the effective kernel scale.
 
    Forward input shape:  (..., input_dim)
    Forward output shape: (..., embedding_dim)
 
    Forward mask contract: ``mask`` (if provided) affects only InputNorm
    statistics.  The forward output is computed for *every* input position
    including padded ones; downstream consumers are responsible for masking
    padded entries before pooling.
    """
 
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        kernel: KernelName = "gaussian",
        sigma: float | None = None,
        matern_nu: float = 1.5,
        riesz_eps: float = 0.1,
        seed: int | None = None,
        normalize_inputs: bool = False,
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
            if kernel == "riesz":
                sigma = 1.0  # Riesz-specific default: set the Cauchy proposal scale to 1.0
            else:
                sigma = math.sqrt(input_dim)  # default heuristic: match typical ||x-y|| scale
        elif sigma <= 0:
            raise ValueError(f"sigma must be > 0 (got {sigma}).")
 
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_freq = embedding_dim // 2
        self.kernel = kernel
        self.sigma = sigma
        self.matern_nu = matern_nu
        self.riesz_eps = riesz_eps
        self.in_norm = InputNorm(input_dim, skip=not normalize_inputs)
 
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(seed)
 
        # Default importance weights: identity (used for all PD kernels).
        sqrt_w = torch.ones(self.num_freq)
 
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
        elif kernel == "riesz":
            # CPD kernel: importance sampling.  sigma is reinterpreted as
            # the 1D Cauchy proposal scale sigma_xi; riesz_eps regularizes
            # the target weight 1/(xi^2 + eps^2).
            omega, sqrt_w = _sample_sliced_riesz(
                self.num_freq, input_dim,
                sigma_xi=sigma, eps=riesz_eps,
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
                "gaussian, laplace, matern, riesz, train."
            )
 
        # Frequencies: learnable when kernel='train', frozen buffer otherwise.
        # Importance weights are always frozen (= 1 except for 'riesz').
        if kernel == "train":
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega)
        self.register_buffer("sqrt_w", sqrt_w)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.shape[-1]}."
            )
        if self.training:
            self.in_norm.update_stats(x, mask=mask)
        x = self.in_norm(x)
        # Linear projection: (..., input_dim) @ (input_dim, num_freq) -> (..., num_freq).
        proj = x @ self.omega.T
        # Apply importance weights uniformly (sqrt_w == 1 for PD kernels, so
        # this is a no-op except for 'riesz').  Keeping a single code path
        # avoids divergent behavior between PD and CPD cases.
        cos_part = torch.cos(proj) * self.sqrt_w
        sin_part = torch.sin(proj) * self.sqrt_w
        out = math.sqrt(2.0) * torch.stack([cos_part, sin_part], dim=-1).flatten(start_dim=-2)
        return out