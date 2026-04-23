import torch
import torch.nn as nn
import torch.nn.functional as F
import torchkit.pytorch_utils as ptu
from torchkit.networks import gpt_like_Mlp


class MateLinAttn(nn.Module):
    name = "mate_linattn"

    def __init__(self, input_size, hidden_size, n_layer, feature_map="elu", pdrop=0.1, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feature_map = feature_map
        self.embedder = gpt_like_Mlp(hidden_size=hidden_size, n_layer=n_layer, pdrop=pdrop)
        self.key_net = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(pdrop))
        self.value_net = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(pdrop))
        self.query_net = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(pdrop))
        self.init_emb = nn.Parameter(ptu.randn(hidden_size))

    def _phi(self, x):
        if self.feature_map == "elu":
            return F.elu(x) + 1
        raise NotImplementedError(f"Feature map {self.feature_map}")

    def forward(self, inputs, h_0, obs_emb, **kwargs):
        """
        inputs: (T, B, input_size)
        h_0: (S_0, z_0) — S_0: (1, B, d, d), z_0: (1, B, d)
        obs_emb: (T, B, hidden_size) — obs embedding for query
        Returns:
        output: (T, B, d)
        h_n: (S_n, z_n)
        info: dict
        """
        S_0, z_0 = h_0
        S_0 = S_0.squeeze(0)  # (B, d, d)
        z_0 = z_0.squeeze(0)  # (B, d)

        x = self.embedder(inputs)
        k = self._phi(self.key_net(x))  # (T, B, d)
        v = self.value_net(x)           # (T, B, d)

        q_input = obs_emb
        q = self._phi(self.query_net(q_input))  # (T, B, d)

        kv = k.unsqueeze(-1) * v.unsqueeze(-2)           # (T, B, d, d)
        S_all = S_0.unsqueeze(0) + kv.cumsum(dim=0)      # (T, B, d, d)
        z_all = z_0.unsqueeze(0) + k.cumsum(dim=0)       # (T, B, d)

        num = torch.einsum('tbij,tbi->tbj', S_all, q) + self.init_emb  # (T, B, d)
        den = (z_all * q).sum(-1, keepdim=True).clamp(min=1e-6)        # (T, B, 1)
        output = num / den  # (T, B, d)

        h_n = (S_all[-1].unsqueeze(0), z_all[-1].unsqueeze(0))
        info = {
            "denominator_mean": den.detach().squeeze(-1).mean(dim=1),
            "denominator_std": den.detach().squeeze(-1).std(dim=1),
        }
        return output, h_n, info

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        S = ptu.zeros((1, batch_size, self.hidden_size, self.hidden_size)).float()
        z = ptu.zeros((1, batch_size, self.hidden_size)).float()
        return S, z

    def internal_state_to_hidden(self, internal_state):
        S, z = internal_state
        return S.mean(dim=-1)  # (1, B, d)
