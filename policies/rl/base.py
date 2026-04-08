from typing import Any, Tuple
from policies.models.actor import MarkovPolicyBase


class RLAlgorithmBase:
    name: str
    continuous_action: bool
    use_target_actor: bool

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes) -> MarkovPolicyBase:
        raise NotImplementedError

    @staticmethod
    def build_critic(input_size, hidden_sizes, **kwargs):
        raise NotImplementedError

    def select_action(self, actor_or_critic, observ, deterministic: bool) -> Any:
        raise NotImplementedError

    @staticmethod
    def forward_actor(actor, observ) -> Tuple[Any, Any]:
        raise NotImplementedError

    def update_others(self, **kwargs):
        pass
