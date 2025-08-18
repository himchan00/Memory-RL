from .policy_rnn_shared import ModelFreeOffPolicy_Shared_RNN as Policy_Shared_RNN
from .policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN as Policy_DQN_RNN
from .policy_rnn_ppo import ModelFreePPO_Shared_RNN as Policy_PPO_RNN

AGENT_CLASSES = {
    "Policy_Shared_RNN": Policy_Shared_RNN,
    "Policy_DQN_RNN": Policy_DQN_RNN,
    "Policy_PPO_RNN": Policy_PPO_RNN
}
