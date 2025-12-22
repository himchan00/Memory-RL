from .dqn import DQN
from .sac import SAC
from .ppo import PPO

RL_ALGORITHMS = {
    DQN.name: DQN,
    SAC.name: SAC,
    PPO.name: PPO
}
