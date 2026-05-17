import torch
import torch.nn as nn


class PopArt(nn.Module):
    """Single-gamma port of amago.nets.actor_critic.PopArtLayer.

    Statistics (mu, nu) are tracked as scalars over the running TD-target
    distribution. The final linear-equivalent (w, b) is corrected after every
    stats update so the unnormalized output is preserved (POP).
    Paper: https://arxiv.org/abs/1809.04474
    """

    def __init__(
        self,
        beta: float = 5e-4,
        init_nu: float = 100.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("nu", torch.ones(1) * init_nu)
        self.register_buffer("w", torch.ones(1))
        self.register_buffer("b", torch.zeros(1))
        self.register_buffer("_t", torch.ones(1))
        self.beta = beta
        self.enabled = enabled

    @property
    def sigma(self) -> torch.Tensor:
        inner = (self.nu - self.mu**2).clamp(1e-4, 1e8)
        return torch.sqrt(inner).clamp(1e-4, 1e6)

    def normalize_values(self, val: torch.Tensor) -> torch.Tensor:
        """Get normalized (Q) values"""
        if not self.enabled:
            return val
        return ((val - self.mu) / self.sigma).to(val.dtype)

    def update_stats(self, val: torch.Tensor, mask: torch.Tensor) -> None:
        """Update the moving average statistics.

        Args:
            val: The value estimate.
            mask: A mask that is 0 where value estimates should be ignored (e.g., from padded timesteps).
        """
        if not self.enabled:
            return
        assert val.shape == mask.shape
        self._t += 1
        old_sigma = self.sigma.data.clone()
        old_mu = self.mu.data.clone()
        # Use adaptive step size to reduce reliance on initialization (pg 13)
        beta_t = self.beta / (1.0 - (1.0 - self.beta) ** self._t)
        total = mask.sum()
        mean = (val * mask).sum() / total
        square_mean = ((val * mask) ** 2).sum() / total
        self.mu.data = (1.0 - beta_t) * self.mu + beta_t * mean
        self.nu.data = (1.0 - beta_t) * self.nu + beta_t * square_mean
        self.w.data *= old_sigma / self.sigma
        self.b.data = ((old_sigma * self.b) + old_mu - self.mu) / (self.sigma)

    def forward(self, x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
        """Modify the value estimate according to the PopArt layer.

        Applies normalization or denormalization to value estimates using PopArt's moving average statistics.
        When normalized=True, scales and shifts values using the current statistics to normalize them.
        When normalized=False, maps normalized values back to the original scale of the environment.

        Args:
            x: Value estimate to modify
            normalized: Whether to normalize (True) or denormalize (False) the values

        Returns:
            Modified value estimate in either normalized or denormalized form
        """
        if not self.enabled:
            return x
        normalized_out = (self.w * x) + self.b
        if normalized:
            return normalized_out.to(x.dtype)
        else:
            return ((self.sigma * normalized_out) + self.mu).to(x.dtype)
