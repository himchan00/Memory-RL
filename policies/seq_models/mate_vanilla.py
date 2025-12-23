import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, obs_dim, n_layer, max_seq_length, pdrop, norm, out_act = "linear", **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.transition_dropout_mask = None
        self.is_target = False
        self.init_emb_mode = kwargs["init_emb_mode"]
        self.init_embedder = init_embedder(obs_dim, kwargs["init_emb_mode"], n_layer, hidden_size, out_act, norm=norm, dropout=pdrop)
        self.embedder = Mlp(hidden_sizes=[4*hidden_size]*(n_layer-1), # one layer is used in transition embedder
                            output_size=hidden_size, input_size=input_size, output_activation= out_act, norm = norm, dropout = pdrop)
        print(f"Use Mate with init_emb_mode = {self.init_emb_mode}")

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size), int 
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size), int 
        """
        L = inputs.shape[0]
        B = inputs.shape[1]
        (hidden, t) = h_0
        mask = self.transition_dropout_mask
        if mask is None: # inference or no dropout
            mask = ptu.ones(L, B)
        if self.is_target: # for calculating target. Use current transitions with dropedout previous transitions
            t_expanded = (t + 1 + torch.cat((ptu.zeros(1, B), mask[:-1]), dim = 0).cumsum(dim = 0)).long() # (L, B)
        else:
            t_expanded = (t + mask.cumsum(dim = 0)).long() # (L, B)
        
        if self.is_target:
            z = self.embedder(inputs) # (L, B, hidden_size)
            z_orig = z.clone()
            z = z * mask.unsqueeze(-1) # (L, B, hidden_size)
            cumsum = hidden * t + z_orig + torch.cat((ptu.zeros(1, B, self.hidden_size), z[:-1]), dim = 0).cumsum(dim=0) # (L, B, hidden_size)
        else:
            z_partial = self.embedder(inputs[mask.bool()]) 
            z = ptu.zeros(L, B, self.hidden_size) # (L, B, hidden_size)
            z[mask.bool()] = z_partial
            cumsum = hidden * t + z.cumsum(dim = 0) # (L, B, hidden_size)
        output = cumsum / t_expanded.clamp(min = 1).unsqueeze(-1) # when t = 0, output = 0
        h_n = output[-1].clone().unsqueeze(0), t_expanded[-1, 0] # h_n is only used for inference (no transition dropout). Thus we can use t from any batch element.
        return output, h_n

    def get_zero_internal_state(self, batch_size=1, init_obs=None, **kwargs):
        """
        init_obs: (B, obs_dim) or None
        """
        if init_obs is None:
            h = ptu.zeros((1, batch_size, self.hidden_size)).float() # (1, B, hidden_size)
            t = 0
        else:
            h = self.init_embedder(init_obs).unsqueeze(0)  # (1, B, hidden_size)
            t = 1
        return h, t # (h_t, t)



    def sample_transition_dropout_mask(self, length: int, batch_size: int, max_drop: float):
        """
        For each batch element b, sample a dropout rate p_b ~ Uniform(0, max_drop),
        then generate a length-dim mask with Bernoulli(keep_prob = 1 - p_b).

        Returns:
            mask: (length, batch_size) float tensor (0.0 = drop, 1.0 = keep)
        """
        assert (0.0 <= max_drop <= 1.0)

        # Per-batch dropout rate: p_b ~ U(0, max_drop)
        drop_p = ptu.rand(batch_size) * max_drop          # (B,)
        keep_p = 1.0 - drop_p                             # (B,)

        # Sample Bernoulli masks with batch-wise keep probabilities (broadcasted)
        u = ptu.rand(length, batch_size)                  # (L,B)
        mask = (u <= keep_p.unsqueeze(0)).to(torch.int32)                 # (L,B)

        return mask
    

    def internal_state_to_hidden(self, internal_state):
        hidden, t = internal_state
        return hidden


class init_embedder(nn.Module):
    def __init__(self, obs_dim, mode, n_layer, hidden_size, out_activation, **kwargs):
        super().__init__()
        self.mode = mode
        if self.mode == "obs":
            self.embedder = Mlp(
                hidden_sizes=n_layer * [4 * hidden_size],
                input_size=obs_dim,
                output_size=hidden_size,
                output_activation=out_activation, # activation applied later
                **kwargs
            )
        elif self.mode == "parameter":
            self.embedding = nn.Parameter(ptu.randn(hidden_size))
        elif self.mode == "zero":
            self.embedding = ptu.zeros(hidden_size)
        else:
            raise NotImplementedError
    
    def forward(self, observs):
        """
        observs: (B, obs_dim)
        return: (B, hidden_size)
        """
        if self.mode == "obs":
            x = self.embedder(observs)
        else:
            bs = observs.shape[0]
            x = self.embedding.reshape(1, -1).repeat(bs, 1)
        return x