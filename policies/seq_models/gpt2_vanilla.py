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
        dropout_emb,
        dropout_ff,
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
            attn_pdrop=0.0,
            resid_pdrop=dropout_ff,
            embd_pdrop=dropout_emb,
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

    def forward(self, input_embeds, h_0, weights=None, **kwargs):
        """
        input_embeds:
            training -- (max_seq_length, B, input_dim)
            eval -- (1, 1, input_dim)
        weights: optional (T, B, 1) in {0, 1}. A 0 hides that position as a KEY,
            so no query attends to it -- the transformer never reads it. The
            position still produces its own output, because the critic needs a
            Q value at every timestep; what it must not do is enter anyone
            else's context. Used for Alchemy's NO_OP steps, which carry nothing
            about the chemistry.
        """
        if h_0 is None:  # training: entire sequence as input
            length = input_embeds.shape[0]
            timesteps = ptu.arange(0, length)
            pkv = None
            keep = self._key_mask(weights)
            attn = self._additive_mask(keep, input_embeds.shape[0])
            output, full_out = self._forward(input_embeds, timesteps, pkv, attn)
            h = full_out["past_key_values"], None, None, output

        else:  # inference/testing: one time step at a time
            pkv, timesteps, past_embeds, _, past_mask = h_0
            history_length = past_embeds.shape[0]
            cur_mask = self._key_mask(weights)      # (B, 1) or None
            if cur_mask is None and past_mask is not None:
                cur_mask = past_mask.new_ones((past_mask.shape[0], 1))
            if history_length > self.max_history_length:  # confirmed this is correct
                pkv = None
                timesteps = ptu.arange(
                    0, self.max_history_length + 1
                )  # match the training
                cur_input_embed = input_embeds

                input_embeds = torch.cat(
                    (past_embeds[-self.max_history_length :], cur_input_embed), dim=0
                )
                if past_mask is not None:
                    attn = torch.cat(
                        (past_mask[:, -self.max_history_length :], cur_mask), dim=1
                    )
                else:
                    attn = None
            else:
                # With a kv-cache the mask must still cover the WHOLE context:
                # past keys live in pkv, not in input_embeds.
                attn = (
                    torch.cat((past_mask, cur_mask), dim=1)
                    if past_mask is not None else None
                )

            output, full_out = self._forward(
                input_embeds, timesteps, pkv,
                self._additive_mask(attn, input_embeds.shape[0]),
            )
            output = output[[-1]]  # (1, 1, hidden_size)
            past_embeds = (
                input_embeds
                if input_embeds.shape[0] > 1
                else torch.cat((past_embeds, input_embeds), dim=0)
            )
            if past_mask is not None:
                past_mask = (
                    attn if input_embeds.shape[0] > 1
                    else torch.cat((past_mask, cur_mask), dim=1)
                )
            h = (full_out["past_key_values"], timesteps + 1, past_embeds,
                 output, past_mask)

        return output, h

    @staticmethod
    def _key_mask(weights):
        """(T, B, 1) in {0, 1} -> (B, T) keep-mask over keys, or None."""
        if weights is None:
            return None
        return weights.squeeze(-1).transpose(0, 1)

    @staticmethod
    def _additive_mask(keep, n_query):
        """(B, K) keep-mask -> (B, 1, n_query, K) additive mask.

        A position hidden as a key must still be visible to ITSELF: it has to
        produce its own output, and the critic needs a Q value there. A 2D key
        mask cannot say that, and getting it wrong is not a small error -- with
        every causally-visible key masked, the causal fill and the mask fill are
        both -1e4, so the softmax spreads onto FUTURE keys. Measured: 1.7e-2
        divergence between the batched and step-by-step paths, entirely at the
        one position that masked itself.

        Queries are the LAST n_query keys, which is how both the full-sequence
        pass (n_query == K) and the kv-cached single step (n_query == 1) line up.
        """
        if keep is None:
            return None
        batch, n_key = keep.shape
        m = keep[:, None, :].expand(batch, n_query, n_key).clone()
        q_pos = torch.arange(n_key - n_query, n_key, device=keep.device)
        m[:, torch.arange(n_query, device=keep.device), q_pos] = 1.0
        return ((1.0 - m) * -10000.0).unsqueeze(1)

    def _forward(self, input_embeds, timesteps, pkv, attention_mask=None):
        """
        input_embeds: (T, B, hidden_size)
        timesteps: (T,)
        pkv: past_key_values
        attention_mask: (B, past + T) with 0 on keys nobody may attend to
        """
        length = timesteps.shape[0]
        pe = self.embed_timestep(timesteps).view(
            length, 1, self.hidden_size
        )  # (T, 1, hidden_size)
        input_embeds_pe = input_embeds + pe
        input_embeds_pe = torch.swapaxes(input_embeds_pe, 0, 1)  # (B, T, hidden_size)
        out = self.transformer(
            inputs_embeds=input_embeds_pe, output_attentions=False, past_key_values=pkv,
            attention_mask=attention_mask,
        )
        last_hidden_state = torch.swapaxes(
            out["last_hidden_state"], 0, 1
        )  # (T, B, hidden_size)

        return last_hidden_state, out

    def get_zero_internal_state(self, batch_size=1, training = False, **kwargs):
        """
        returns None if training, else (pkv=None, timestep=(1,), past_embeds=(0, B, H), output=(0, B, H), key_mask=(B, 0))
        """
        if training:
            return None
        else:
            pkv = None
            initial_timestep = ptu.arange(0, 1)  # (1,)  Assumption: batch is synchronized
            return (
                pkv, initial_timestep,
                ptu.zeros((0, batch_size, self.hidden_size)).float(),
                ptu.zeros((0, batch_size, self.hidden_size)).float(),
                ptu.zeros((batch_size, 0)).float(),   # key mask over the history
            )

    def internal_state_to_hidden(self, internal_state):
        pkv, timesteps, past_embeds, output = internal_state
        return output