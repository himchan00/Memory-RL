from .sacd import SACD
from .dqn import DQN
from .sac import SAC

RL_ALGORITHMS = {
    SACD.name: SACD,
    DQN.name: DQN,
    SAC.name: SAC,
}
