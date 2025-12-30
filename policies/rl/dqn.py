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
        # clamp step to [0, n]
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
        qf = FlattenMlp(
            input_size=input_size, output_size=action_dim, hidden_sizes=hidden_sizes
        )
        return qf

    def select_action(self, qf, observ, deterministic: bool):
        bs = observ.shape[0]
        action_logits = qf(observ)  # (B=1, A)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)  # (*)
        else:
            random_action = torch.randint(
                high=action_logits.shape[-1], size=action_logits.shape[:-1]
            ).to(
                ptu.device
            )  # (*)
            optimal_action = torch.argmax(action_logits, dim=-1)  # (*)

            eps = self.epsilon_schedule(self.count)
            # mask = 0 means 1-eps exploit; mask = 1 means eps explore
            mask = torch.multinomial(
                input=ptu.FloatTensor([1 - eps, eps]),
                num_samples=action_logits.shape[0],
                replacement=True,
            )  # (*)
            action = mask * random_action + (1 - mask) * optimal_action

            self.count += bs
            # print(eps, self.count, random_action, optimal_action, action)

        # convert to one-hot vectors
        action = F.one_hot(
            action.long(), num_classes=action_logits.shape[-1]
        ).float()  # (*, A)
        return action

    def critic_loss(
        self,
        critic,
        critic_target,
        observs,
        actions,
        rewards,
        terms,
        gamma,
    ):
        # Q^tar(h(t+1), pi(h(t+1))) + H[pi(h(t+1))]
        with torch.no_grad():
            if critic.head.seq_model.name == "mate":
                critic.head.is_target = True
            next_v, _ = critic(
                actions=actions,
                rewards=rewards,
                observs=observs
            )  # (T+1, B, A)
            if critic.head.seq_model.name == "mate":
                critic.head.is_target = False
            next_target_v, _ = critic_target(
                actions=actions,
                rewards=rewards,
                observs=observs
            )  # (T+1, B, A)

            next_actions = torch.argmax(next_v, dim=-1, keepdim=True)  # (*, 1)
            next_target_q = next_target_v.gather(dim=-1, index=next_actions)  # (*, 1)

            next_target_q = next_target_q[1:]
            q_target = rewards + (1.0 - terms) * gamma * next_target_q  # next q

        # Q(h(t), a(t)) (T, B, 1)
        v_pred, d_loss = critic(
            actions=actions,
            rewards=rewards,
            observs=observs
        )  # (T, B, A)
        v_pred = v_pred[:-1]

        actions = torch.argmax(
            actions, dim=-1, keepdim=True
        )  # (T, B, 1)
        q_pred = v_pred.gather(
            dim=-1, index=actions
        )  # (T, B, A) -> (T, B, 1)

        return q_pred, q_target, d_loss
