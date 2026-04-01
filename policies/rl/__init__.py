from .dqn import DQN
from .sac import SAC

RL_ALGORITHMS = {
    DQN.name: DQN,
    SAC.name: SAC
}
