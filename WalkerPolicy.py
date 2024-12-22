import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta

DIM_ACTOR_NN = 256  # Increased size
DIM_VALUE_NN = 128  # Increased size

class WalkerPolicy(nn.Module):
    def __init__(self, state_dim: int = 29, action_dim: int = 8, load_weights: bool = False):
        super().__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(state_dim, DIM_ACTOR_NN),
            nn.ReLU(),
            nn.Linear(DIM_ACTOR_NN, DIM_ACTOR_NN),
            nn.ReLU(),
            nn.Linear(DIM_ACTOR_NN, action_dim * 2)  # Output alpha and beta parameters for Beta distribution
        )
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, DIM_VALUE_NN),
            nn.ReLU(),
            nn.Linear(DIM_VALUE_NN, DIM_VALUE_NN),
            nn.ReLU(),
            nn.Linear(DIM_VALUE_NN, 1)
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
            alpha, beta = torch.chunk(mean, 2, dim=-1)
            alpha = F.softplus(alpha) + 1
            beta = F.softplus(beta) + 1
            dist = Beta(alpha, beta)
            actions = dist.sample()
        return actions

    def forward(self, states: torch.Tensor):
        """
        Given states tensor, returns the actions, log probabilities, and state values.
        This is used for training the agent.

        Args:
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            actions (torch.Tensor): (N, action_dim) tensor
            log_probs (torch.Tensor): (N, action_dim) tensor
            state_values (torch.Tensor): (N, 1) tensor
        """
        mean = self.actor_network(states)
        alpha, beta = torch.chunk(mean, 2, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        dist = Beta(alpha, beta)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        state_values = self.value_network(states)
        return actions, log_probs, state_values

    def log_prob(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Given actions and states tensors, returns the log probabilities of the actions.
        This is used for computing the PPO loss.

        Args:
            actions (torch.Tensor): (N, action_dim) tensor
            states (torch.Tensor): (N, state_dim) tensor

        Returns:
            log_probs (torch.Tensor): (N, action_dim) tensor
        """
        mean = self.actor_network(states)
        alpha, beta = torch.chunk(mean, 2, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        dist = Beta(alpha, beta)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return log_probs

    def save_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to save your network weights
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str = 'walker_weights.pt') -> None:
        # helper function to load your network weights
        self.load_state_dict(torch.load(path))