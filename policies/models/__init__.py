from .policy_rnn_sac import ModelFreeOffPolicy_SAC_RNN as Policy_SAC_RNN
from .policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN as Policy_DQN_RNN

AGENT_CLASSES = {
    "Policy_SAC_RNN": Policy_SAC_RNN,
    "Policy_DQN_RNN": Policy_DQN_RNN
}
