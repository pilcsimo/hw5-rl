import torch


def policy_gradient_loss_simple(
    logp: torch.Tensor, tensor_r: torch.Tensor
) -> torch.Tensor:
    """
    Given the log-probabilities of the policy and the rewards, compute the scalar loss
    representing the policy gradient.

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        tensor_r: (T, N) tensor of rewards, detached from any computation graph

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: start by calculating the cumulative returns of the trajectories, and then compute the policy gradient
    T, N = logp.shape  # T is the episode length, N is the number of trajectories
    grad_sum = torch.sum(logp, dim=0)
    g_hat = torch.sum(tensor_r * grad_sum) / (T * N)
    return -g_hat


def discount_cum_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Given the rewards and the discount factor gamma, compute the discounted cumulative sum of rewards.
    The cumulative sum follows the reward-to-go formulation. This means we want to compute the discounted
    trajectory returns at each timestep. We do that by calculating an exponentially weighted
    sum of (only) the following rewards.
    i.e.
    $R(\tau_i, t) = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'}$

    Args:
        rewards: (T, N) tensor of rewards
        gamma: discount factor

    Returns:
        discounted_cumulative_returns: (T, N) tensor of discounted cumulative returns
    """
    # TODO: implement the discounted cummulative sum, i.e. the discounted returns computed from rewards and gamma
    T, N = rewards.shape
    discounted_cumulative_returns = torch.zeros_like(rewards)
    for t in reversed(range(T)):
        if t == T - 1:
            discounted_cumulative_returns[t] = rewards[t]
        else:
            discounted_cumulative_returns[t] = (
                rewards[t] + gamma * discounted_cumulative_returns[t + 1]
            )
    return discounted_cumulative_returns


def policy_gradient_loss_discounted(
    logp: torch.Tensor, tensor_r: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Given the policy log0probabilities, rewards and the discount factor gamma, compute the
    policy gradient loss using discounted returns.

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        tensor_r: (T, N) tensor of rewards, detached from any computation graph
        gamma: discount factor

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: compute discounted returns of the trajectories from the reward tensor, then compute the policy gradient
    T, N = logp.shape  # T is the episode length, N is the number of trajectories
    with torch.no_grad():
        discounted_returns = discount_cum_sum(tensor_r, gamma)
    # Compute the policy gradient loss using the discounted returns
    # NOTE: needs to be able to be backpropagated-through
    g_hat = torch.sum(discounted_returns * logp) / (T * N)
    return -g_hat


def policy_gradient_loss_advantages(
    logp: torch.Tensor, advantage_estimates: torch.Tensor
) -> torch.Tensor:
    """
    Given the policy log-probabilities and the advantage estimates, compute the policy gradient loss

    Args:
        logp: (T, N) tensor of log-probabilities of the policy
        advantage_estimates: (T, N) tensor of advantage estimates

    Returns:
        policy_loss: scalar tensor representing the policy gradient loss
    """
    # TODO: compute the policy gradient estimate using the advantage estimate weighting
    T, N = logp.shape  # T is the episode length, N is the number of trajectories
    with torch.no_grad():
        g_hat = torch.sum(logp * advantage_estimates) / (T * N)
    return -g_hat


def value_loss(values: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    """
    Given the values and the value targets, compute the value function regression loss
    """
    # TODO: compute the value function L2 loss
    T, N = values.shape  # T is the episode length, N is the number of trajectories
    # sum of L2 losses
    l2_loss = torch.sum((values - value_targets) ** 2) / (T * N)
    return l2_loss


def ppo_loss(p_ratios, advantage_estimates, epsilon):
    """
    Given the probability ratios, advantage estimates and the clipping parameter epsilon, compute the PPO loss
    based on the clipped surrogate objective
    """
    # TODO: compute the PPO loss
    T, N = p_ratios.shape  # T is the episode length, N is the number of trajectories
    # Compute the clipped surrogate objective
    clipped_surrogate = torch.min(
        p_ratios * advantage_estimates,
        torch.clamp(p_ratios, 1 - epsilon, 1 + epsilon) * advantage_estimates,
    )
    # Compute the PPO loss
    ppo_loss = -torch.sum(clipped_surrogate) / (T * N)
    return ppo_loss
