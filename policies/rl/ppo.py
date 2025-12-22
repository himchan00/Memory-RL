from policies.models.actor import TanhGaussianPolicy
from torchkit.networks import FlattenMlp



class PPO():
    name = "ppo"

    def __init__(
        self, **kwargs
    ):
        pass


    @staticmethod
    def build_actor(input_size, action_dim, hidden_sizes, **kwargs):
        return TanhGaussianPolicy(
            obs_dim=input_size,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            **kwargs,
        )


    @staticmethod
    def build_critic(hidden_sizes, input_size):

        v = FlattenMlp(
            input_size=input_size, output_size=1, hidden_sizes=hidden_sizes
        )

        return v


