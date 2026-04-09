import torch
from .base import RLAlgorithmBase
from torchkit.networks import FlattenMlp
import torch.nn.functional as F
import torchkit.pytorch_utils as ptu


class LinearSchedule:
    def __init__(self, init_value: float, end_value: float, transition_steps: int):
        self.init = float(init_value)
        self.end = float(end_value)
        self.n = max(1, int(transition_steps))

    def __call__(self, step: int) -> float:
        t = 0 if step < 0 else self.n if step > self.n else step
        frac = t / self.n
        return (1.0 - frac) * self.init + frac * self.end


class DQN(RLAlgorithmBase):
    name = "dqn"
    continuous_action = False

    def __init__(self, init_eps=1.0, end_eps=0.01, schedule_steps=1000, **kwargs):
        self.epsilon_schedule = LinearSchedule(
            init_value=init_eps,
            end_value=end_eps,
            transition_steps=schedule_steps,
        )
        self.count = 0

    @staticmethod
    def build_critic(hidden_sizes, input_size=None, obs_dim=None, action_dim=None):
        if obs_dim is not None and action_dim is not None:
            input_size = obs_dim + action_dim
        return FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )

    def select_action(self, qf, observ, deterministic: bool):
        bs = observ.shape[0]
        action_logits = qf(observ)  # (B, A)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            random_action = torch.randint(
                high=action_logits.shape[-1], size=action_logits.shape[:-1]
            ).to(ptu.device)
            optimal_action = torch.argmax(action_logits, dim=-1)

            eps = self.epsilon_schedule(self.count)
            mask = torch.multinomial(
                input=ptu.FloatTensor([1 - eps, eps]),
                num_samples=action_logits.shape[0],
                replacement=True,
            )
            action = mask * random_action + (1 - mask) * optimal_action
            self.count += bs

        return F.one_hot(action.long(), num_classes=action_logits.shape[-1]).float()

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state_dict):
        self.count = state_dict["count"]
