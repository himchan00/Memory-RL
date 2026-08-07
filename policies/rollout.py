from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EpisodeTrajectory:
    observations: list[torch.Tensor]
    actions: list[torch.Tensor]
    rewards: list[torch.Tensor]
    next_observations: list[torch.Tensor]
    terminals: list[torch.Tensor]
    memory_masks: list[torch.Tensor]

    @classmethod
    def start(
        cls,
        *,
        prev_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        obs: torch.Tensor,
        terminal: torch.Tensor,
        memory_mask: torch.Tensor,
    ):
        return cls(
            observations=[prev_obs],
            actions=[action],
            rewards=[reward],
            next_observations=[obs],
            terminals=[terminal],
            memory_masks=[memory_mask],
        )

    def append(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        terminal: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> None:
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_obs)
        self.terminals.append(terminal)
        self.memory_masks.append(memory_mask)

    def commit(self, buffer, *, continuous_actions: bool) -> torch.Tensor:
        actions = torch.stack(self.actions, dim=0)
        if not continuous_actions:
            actions = torch.argmax(actions, dim=-1, keepdim=True)

        rewards = torch.stack(self.rewards, dim=0)
        buffer.add_episode(
            actions=actions,
            observations=torch.stack(self.observations, dim=0),
            next_observations=torch.stack(self.next_observations, dim=0),
            rewards=rewards,
            terminals=torch.stack(self.terminals, dim=0),
            memory_masks=torch.stack(self.memory_masks, dim=0),
        )
        return rewards


@dataclass
class RolloutResult:
    episode_return: float
    success_rate: float
    average_steps: float
    rewards: torch.Tensor | None
    frames: list[np.ndarray] | None
    attempt_returns: np.ndarray | None = None
    attempt_success: np.ndarray | None = None

