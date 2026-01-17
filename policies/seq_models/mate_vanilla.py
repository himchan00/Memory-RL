import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu
from torchkit.networks import Mlp, gpt_like_Mlp


class Mate(nn.Module):
    name = "mate"

    def __init__(self, input_size, hidden_size, n_layer, max_seq_length, pdrop, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.embedder = gpt_like_Mlp(hidden_size=hidden_size, n_layer=n_layer, pdrop=pdrop, use_output_ln=False)
        # self.embedder = Mlp(hidden_sizes=[hidden_size]*(n_layer-1), # one layer is used in transition embedder
        #                     output_size=hidden_size, input_size=input_size, output_activation= out_act, norm = norm, dropout = pdrop)

    def forward(self, inputs, h_0):
        """
        inputs: (T, B, hidden_size)
        h_0: (1, B, hidden_size)
        return
        output: (T, B, hidden_size)
        h_n: (1, B, hidden_size)
        """
        z = self.embedder(inputs) # (L, B, hidden_size)
        cumsum = h_0 + z.cumsum(dim = 0) # (L, B, hidden_size)
        output = cumsum
        h_n = output[-1].clone().unsqueeze(0)
        return output, h_n

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        """
        init_obs: (B, obs_dim) or None
        """
        return ptu.zeros((1, batch_size, self.hidden_size)).float() # (1, B, hidden_size)



    def sample_permutation_indices(self, episode_length: int, batch_size: int):
        """

        Returns:
            memory_permutation: (episode_length+1, batch_size) long tensor
        """
        memory_perm = ptu.rand(episode_length+1, batch_size).argsort(dim=0)  # (L+1, B)
        return memory_perm
    

    def internal_state_to_hidden(self, internal_state):
        return internal_state  # identity mapping for Mate
