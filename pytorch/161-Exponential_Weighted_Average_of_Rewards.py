import torch

def exp_weighted_average(Q1: float, rewards: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Q1: float, initial estimate
    rewards: torch.Tensor of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: torch.Tensor (scalar), exponentially weighted average after k rewards
    """
    # Your code here
    sigma = 0
    k = len(rewards)
    i = 1
    for reward in rewards:
        sum = alpha * (1 - alpha)**(k-i) * reward
        sigma += sum
        i += 1

    result = ((1 - alpha)**k * Q1) + sigma
    return result