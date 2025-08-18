from policies.models.actor import TanhGaussianPolicy, CategoricalPolicy
from torchkit.networks import FlattenMlp



class PPO():
    name = "ppo"

    def __init__(
        self, continuous_action, **kwargs
    ):
        self.continuous_action = continuous_action


    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, continuous_action, **kwargs):
        if continuous_action:
            return TanhGaussianPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )
        else:
            return CategoricalPolicy(
                obs_dim=input_size,
                action_dim=action_dim,
                hidden_sizes=hidden_sizes,
                **kwargs,
            )

    @staticmethod
    def build_critic(hidden_sizes, input_size=None):

        v = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )

        return v


