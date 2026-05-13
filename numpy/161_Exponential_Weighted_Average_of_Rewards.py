def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
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

Q1 = 2.0
rewards = [5.0, 9.0]
alpha = 0.3
result = exp_weighted_average(Q1, rewards, alpha)
print(round(result, 4))