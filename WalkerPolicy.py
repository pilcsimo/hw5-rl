import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class WalkerPolicy(nn.Module):
    """
    A simpler actor-critic model for your quadruped using a Normal distribution.
    This is similar to the pendulum example's approach:
      - We have an actor network that outputs (mu, sigma) for each action dim.
      - We have a critic network that outputs a scalar value estimate V(s).
    """

    def __init__(self, load_weights=False,
                 state_dim=29, 
                 action_dim=8, 
                 hidden_actor_sizes=(64,128,64),
                 hidden_critic_sizes=(64,128,64)):
        super().__init__()
        
        # Custom name for this model
        self.name = "Freddy"
        
        # ----------------------
        # Actor Network
        # ----------------------
        actor_layers = []
        input_size = state_dim
        # build hidden layers
        for hidden_size in hidden_actor_sizes:
            actor_layers.append(nn.Linear(input_size, hidden_size))
            actor_layers.append(nn.ReLU())
            input_size = hidden_size
        # final layer -> 2 * action_dim (mu, sigma)
        actor_layers.append(nn.Linear(input_size, 2 * action_dim))
        self.actor_network = nn.Sequential(*actor_layers)

        # ----------------------
        # Critic Network
        # ----------------------
        critic_layers = []
        input_size = state_dim
        for hidden_size in hidden_critic_sizes:
            critic_layers.append(nn.Linear(input_size, hidden_size))
            critic_layers.append(nn.ReLU())
            input_size = hidden_size
        # final value head -> 1 output
        critic_layers.append(nn.Linear(input_size, 1))
        self.value_network = nn.Sequential(*critic_layers)

        print("WalkerPolicy (Normal) created with:\n"
              f"Actor: {self.actor_network}\n"
              f"Critic: {self.value_network}")
        
        if load_weights:
            self.load_weights()

    def forward(self, states: torch.Tensor):
        """
        We return:
          actions:      [N, action_dim]
          log_probs:    [N]
          state_values: [N, 1]
          (We can skip returning exploration_var now since we dropped Beta.)
        """
        # 1) Actor: get mu, sigma
        actor_out = self.actor_network(states)       # [N, 2*action_dim]
        mu, sigma = torch.chunk(actor_out, 2, dim=-1)   # each [N, action_dim]

        # ensure sigma > 0
        sigma = F.softplus(sigma) + 1e-5  # small offset to avoid zero

        # 2) Create Normal distribution & sample actions
        dist = Normal(mu, sigma)
        actions = dist.sample()                                 # [N, action_dim]
        log_probs = dist.log_prob(actions).sum(dim=-1)          # [N]

        # 3) Critic: value estimate
        state_values = self.value_network(states)               # [N, 1]

        return actions, log_probs, state_values, None

    def determine_actions(self, states: torch.Tensor):
        """
        For deployment: returns a 'deterministic' or 'mean' action (or you can still sample).
        We'll just return mu for a more stable control approach. 
        """
        with torch.no_grad():
            actor_out = self.actor_network(states)              # [N, 2*action_dim]
            mu, sigma = torch.chunk(actor_out, 2, dim=-1)
            sigma = F.softplus(sigma) + 1e-5
            # We'll do deterministic = mu. If you prefer stochastic, sample from Normal.
            return mu  # shape [N, action_dim]

    def log_prob(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Compute log_prob of given actions under the current Normal distribution.
        actions & states shape: [N, action_dim], [N, state_dim].
        """
        actor_out = self.actor_network(states)
        mu, sigma = torch.chunk(actor_out, 2, dim=-1)
        sigma = F.softplus(sigma) + 1e-5
        dist = Normal(mu, sigma)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # [N]
        return log_probs

    def value_estimates(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns V(s) = critic output, shape [N, 1].
        """
        return self.value_network(states)

    def save_weights(self, path='walker_weights.pt'):
        torch.save(self.state_dict(), path)

    def load_weights(self, path='walker_weights.pt'):
        self.load_state_dict(torch.load(path))
