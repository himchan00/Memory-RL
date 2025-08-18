import torchkit.pytorch_utils as ptu
import torch
class RolloutBuffer:
    def __init__(self, observation_dim, action_dim, max_episode_len, n_steps):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.max_episode_len = max_episode_len
        self.n_steps = n_steps
        self.reset()

    
    def add_episode(self, actions, observations, next_observations, logprobs, rewards, values, terminals):
        assert len(actions) == len(observations) == len(next_observations) == len(logprobs) == len(rewards) == len(values) == len(terminals)
        self.actions[:, self.current_steps, :] = actions.detach()
        self.observations[:, self.current_steps, :] = observations.detach()
        self.next_observations[:, self.current_steps, :] = next_observations.detach()
        self.logprobs[:, self.current_steps, :] = logprobs.detach()
        self.rewards[:, self.current_steps, :] = rewards.detach()
        self.values[:, self.current_steps, :] = values.detach()
        self.terminals[:, self.current_steps, :] = terminals.detach()
        self.masks[:len(actions), self.current_steps, :] = 1.0
        self.current_steps += 1

    def reset(self):
        self.actions = ptu.zeros((self.max_episode_len, self.n_steps, self.action_dim))
        self.observations = ptu.zeros((self.max_episode_len, self.n_steps, self.observation_dim))
        self.next_observations = ptu.zeros((self.max_episode_len, self.n_steps, self.observation_dim))
        self.logprobs = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.rewards = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.values = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.terminals = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.masks = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.advantages = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.returns = ptu.zeros((self.max_episode_len, self.n_steps, 1))
        self.current_steps = 0


    def compute_gae(self, gamma, lam):
        """
        Note: this code does not bootstrap final value.
        This is allowed for finite horizon tasks.
        """
        B, T = self.n_steps, self.max_episode_len

        rewards   = self.rewards.squeeze(-1)    # (T, B)
        values    = self.values.squeeze(-1)     # (T, B)
        terminals = self.terminals.squeeze(-1)  # (T, B)  in {0,1}
        
        next_values = torch.zeros_like(values)
        next_values[:-1] = values[1:]      # V_{t+1}

        # TD residuals (delta_t)
        # delta_t = r_t + gamma * (1 - done_t) * V_{t+1} - V_t
        deltas = rewards + gamma * (1.0 - terminals) * next_values - values # (T, B)

        advantages = torch.zeros_like(values)
        gae = ptu.zeros((B,)) 

        for t in reversed(range(T)):
            gae = deltas[t, :] + gamma * lam * (1.0 - terminals[t, :]) * gae   # (B, 1)
            advantages[t, :] = gae

        returns = advantages + values
        self.advantages = advantages.unsqueeze(-1)
        self.returns = returns.unsqueeze(-1)
        return returns, advantages
    