from .markov_vanilla import Markov
from .rnn_vanilla import RNN, LSTM, GRU
from .gpt2_vanilla import GPT2
from .mate_vanilla import Mate


SEQ_MODELS = {Markov.name: Markov, RNN.name: RNN, LSTM.name: LSTM, GRU.name: GRU, GPT2.name: GPT2, Mate.name: Mate}