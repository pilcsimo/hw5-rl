import torch
from torch import nn
from torch.distributions import Normal

class WalkerPolicy(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 8, load_weights: bool = False):
        super().__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # load learned stored network weights after initialization
        if load_weights:
            self.load_weights()

    def determine_actions(self, states: torch.Tensor) -> torch.Tensor:
        """
        Given states tensor, returns the deterministic actions tensor.
        This would be used for control.

        Args:
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            actions (torch.Tensor): (N, action_dim) tensor
        """
        with torch.no_grad():
            mean = self.actor_network(states)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def forward(self, states: torch.Tensor):
        """
        Given states tensor, returns the actions, log probabilities, and state values.
        This would be used for training.

        Args:
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            actions (torch.Tensor): (N, action_dim) tensor
            log_probs (torch.Tensor): (N, action_dim) tensor
            state_values (torch.Tensor): (N, 1) tensor
        """
        mean = self.actor_network(states)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        state_values = self.value_network(states)
        return actions, log_probs, state_values

    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to save your network weights
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to load your network weights
        self.load_state_dict(torch.load(path))