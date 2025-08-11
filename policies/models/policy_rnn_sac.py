# policies/model/policy_rnn_sac.py
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class TanhGaussianPolicy(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, z: torch.Tensor, deterministic: bool = False, with_logprob: bool = True):
        h = self.net(z)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        pre_tanh = mu if deterministic else dist.rsample()
        a = torch.tanh(pre_tanh)
        logp = None
        if with_logprob:
            # Tanh correction term
            logp = dist.log_prob(pre_tanh).sum(-1) - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a, logp, mu

class QNet(nn.Module):
    def __init__(self, z_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, z: torch.Tensor, a: torch.Tensor):
        return self.net(torch.cat([z, a], dim=-1))

class PolicyRNN_SAC(nn.Module):
    """
    Continuous SAC with a shared sequence encoder (RNN/Transformer).
    API matches the discrete version enough that learner changes are minimal.
    """
    def __init__(self, seq_encoder: nn.Module, act_dim: int,
                 shared_encoder: bool = True,
                 fixed_alpha = None,
                 target_entropy = None):
        super().__init__()
        self.shared_encoder = shared_encoder

        # encoders
        self.encoder_actor = seq_encoder
        self.encoder_critic = seq_encoder if shared_encoder else seq_encoder.copy_for_critic()
        z_dim = self.encoder_actor.output_dim

        # heads
        self.actor = TanhGaussianPolicy(z_dim, act_dim)
        self.q1 = QNet(z_dim, act_dim)
        self.q2 = QNet(z_dim, act_dim)
        self.q1_target = QNet(z_dim, act_dim)
        self.q2_target = QNet(z_dim, act_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # temperature α (auto by default)
        if fixed_alpha is None:
            self.log_alpha = nn.Parameter(torch.tensor(0.0))
            self._fixed_alpha = None
        else:
            self.log_alpha = None
            self._fixed_alpha = float(fixed_alpha)

        self.target_entropy = target_entropy  # set later to -act_dim if None

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp() if self.log_alpha is not None else self._fixed_alpha

    @torch.no_grad()
    def act(self, obs_hist: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        z = self.encoder_actor(obs_hist)     # [B, z]
        a, _, _ = self.actor(z, deterministic, with_logprob=False)
        return a

    # convenience used by learner
    def encode_actor(self, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.encoder_actor(obs_hist)

    def encode_critic(self, obs_hist: torch.Tensor) -> torch.Tensor:
        return self.encoder_critic(obs_hist)


#CUrrently not using...........