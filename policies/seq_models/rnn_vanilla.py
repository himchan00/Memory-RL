import torch
import torch.nn as nn
import torchkit.pytorch_utils as ptu


class RNN(nn.Module):
    name = "rnn"
    rnn_class = nn.RNN

    def __init__(self, input_size, hidden_size, n_layer, dropout_ff=0.05, **kwargs):
        super().__init__()
        self.model = self.rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layer,
            batch_first=False,
            dropout=dropout_ff,
            bias=True,
        )
        self.hidden_size = hidden_size
        self.num_layers = n_layer

        self._initialize()

    def _initialize(self):
        # default RNN initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.model.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, inputs, h_0, weights=None, **kwargs):
        """
        inputs: (T, B, input_dim)
        h_0: (num_layers=1, B, hidden_size)
        weights: optional (T, B, 1) in {0, 1}. A 0 means the step does NOT
            advance the state: h_t = h_{t-1}. Used for Alchemy's NO_OP steps,
            which carry nothing about the chemistry.
        return
        output: (T, B, hidden_size)
        h_n: (num_layers=1, B, hidden_size), only used in inference
        """
        if weights is None:
            # cuDNN-fused over the whole sequence; nothing to skip.
            output, h_n = self.model(inputs, h_0)
            return output, h_n

        # A recurrence cannot skip a step without being unrolled: the fused call
        # has no way to say "leave the state alone here". So step through and
        # blend. The output at a skipped step is the carried-over state, which
        # is what the critic should read there -- the step happened in the
        # world, it just added nothing to the memory.
        outputs = []
        h = h_0
        for t in range(inputs.shape[0]):
            out_t, h_new = self.model(inputs[t:t + 1], h)
            w = weights[t].unsqueeze(0)                      # (1, B, 1)
            h = self._blend(h_new, h, w)
            outputs.append(self._state_to_output(h, out_t, w))
        return torch.cat(outputs, dim=0), h

    @staticmethod
    def _blend(new, old, w):
        """w * new + (1 - w) * old, over a tensor state or a tuple of them."""
        if isinstance(new, tuple):
            return tuple(w * n + (1.0 - w) * o for n, o in zip(new, old))
        return w * new + (1.0 - w) * old

    @staticmethod
    def _state_to_output(h, out_t, w):
        """The layer's output for this step, consistent with the blended state.

        nn.RNN/GRU/LSTM return the TOP layer's hidden as the output, so reading
        it off the blended state keeps output and state in lockstep -- which is
        what makes the stepwise rollout reproduce the batched pass exactly.
        """
        top = (h[0] if isinstance(h, tuple) else h)[-1:]      # (1, B, H)
        return top

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        return ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()


class GRU(RNN):
    name = "gru"
    rnn_class = nn.GRU


class LSTM(RNN):
    name = "lstm"
    rnn_class = nn.LSTM

    def get_zero_internal_state(self, batch_size=1, **kwargs):
        # for LSTM, current_internal_state also includes cell state
        hidden_state = ptu.zeros(
            (self.num_layers, batch_size, self.hidden_size)
        ).float()
        cell_state = ptu.zeros((self.num_layers, batch_size, self.hidden_size)).float()
        return hidden_state, cell_state

    def internal_state_to_hidden(self, internal_state):
        hidden_state, cell_state = internal_state
        return hidden_state[-1].unsqueeze(0)  # (1, B, hidden_size)