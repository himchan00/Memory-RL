import torch.nn as nn
import torchkit.pytorch_utils as ptu
import transformers
from .trajectory_gpt2 import GPT2Model
import torch
import numpy as np


class SinePositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size) -> None:
        super().__init__()
        # Create matrix of [max_len, d] representing the positional encoding for max_len inputs
        pe = np.zeros((max_len, hidden_size))
        position = np.arange(0, max_len, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, hidden_size, 2) * (-np.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = ptu.from_numpy(pe)  # (max_len, d)

    def forward(self, timestep):
        return self.pe[timestep]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, hidden_size)

    def forward(self, timestep):
        # (T,)
        return self.pe(timestep)

class DummyPositionalEncoding(nn.Module):
    def __init__(self, max_len, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, timestep):
        # (T,)
        return ptu.zeros((timestep.shape[0], self.hidden_size))

class GPT2(nn.Module):
    name = "gpt"

    def __init__(
        self,
        input_size,
        hidden_size,
        n_layer,
        n_head,
        pdrop,
        max_seq_length,
        position_encoding,
        **kwargs
    ):
        super().__init__()
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter, we don't use word embeddings
            n_layer=n_layer,
            n_head=n_head,
            n_embd=hidden_size,
            attn_pdrop=pdrop,
            resid_pdrop=pdrop,
            embd_pdrop=pdrop,
            # Maximum length sequence the transformer will see; default 1024 might be not long
            n_positions=max_seq_length,
        )  # needs to be divisible by n_head

        self.transformer = GPT2Model(config)

        if position_encoding == "sine":
            Encoding = SinePositionalEncoding
        elif position_encoding == "learned":
            Encoding = LearnedPositionalEncoding
        elif position_encoding == "none":
            Encoding = DummyPositionalEncoding
        else:
            raise NotImplementedError
        self.embed_timestep = Encoding(max_seq_length, hidden_size)

        assert input_size == hidden_size
        self.hidden_size = hidden_size
        self.max_history_length = max_seq_length - 1
        print({k: v.shape for k, v in self.transformer.named_parameters()})

    def forward(self, input_embeds, h_0):
        """
        input_embeds:
            training -- (max_seq_length, B, input_dim)
            eval -- (1, 1, input_dim)
        """
        if h_0 is None:  # training: entire sequence as input
            length = input_embeds.shape[0]
            timesteps = ptu.arange(0, length)
            pkv = None
            output, full_out = self._forward(input_embeds, timesteps, pkv)
            h = full_out["past_key_values"], None, None

        else:  # inference/testing: one time step at a time
            pkv, timesteps, past_embeds = h_0
            history_length = past_embeds.shape[0]
            if history_length > self.max_history_length:  # confirmed this is correct
                pkv = None
                timesteps = ptu.arange(
                    0, self.max_history_length + 1
                )  # match the training
                cur_input_embed = input_embeds

                input_embeds = torch.cat(
                    (past_embeds[-self.max_history_length :], cur_input_embed), dim=0
                )

            output, full_out = self._forward(input_embeds, timesteps, pkv)
            output = output[[-1]]  # (1, 1, hidden_size)
            past_embeds = (
                input_embeds
                if input_embeds.shape[0] > 1
                else torch.cat((past_embeds, input_embeds), dim=0)
            )
            h = full_out["past_key_values"], timesteps + 1, past_embeds
            # print(history_length, self.max_history_length, past_embeds.shape)

        return output, h

    def _forward(self, input_embeds, timesteps, pkv):
        """
        input_embeds: (T, B, hidden_size)
        timesteps: (T,)
        pkv: past_key_values
        """
        length = timesteps.shape[0]
        pe = self.embed_timestep(timesteps).view(
            length, 1, self.hidden_size
        )  # (T, 1, hidden_size)
        input_embeds_pe = input_embeds + pe
        input_embeds_pe = torch.swapaxes(input_embeds_pe, 0, 1)  # (B, T, hidden_size)
        out = self.transformer(
            inputs_embeds=input_embeds_pe, output_attentions=False, past_key_values=pkv
        )
        last_hidden_state = torch.swapaxes(
            out["last_hidden_state"], 0, 1
        )  # (T, B, hidden_size)

        return last_hidden_state, out

    def get_zero_internal_state(self, batch_size=1, training = False, **kwargs):
        """
        returns None if training, else returns (pkv=None, timestep=(1,), past_embeds=(0, B, H)) 
        """
        if training:
            return None
        else:
            pkv = None
            initial_timestep = ptu.arange(0, 1)  # (1,)  Assumption: batch is synchronized
            return pkv, initial_timestep, ptu.zeros((0, batch_size, self.hidden_size)).float()
