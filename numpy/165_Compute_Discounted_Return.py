import numpy as np

def discounted_return(rewards, gamma):
    """
    Compute the total discounted return for a sequence of rewards.
    Args:
        rewards (list or np.ndarray): List or array of rewards [r_0, r_1, ..., r_T-1]
        gamma (float): Discount factor (0 < gamma <= 1)
    Returns:
        float: Total discounted return
    """
    # Your code here
    discounted_return_value = 0
    discount = 1

    for reward in rewards:
        discounted_return_value += (discount * reward)
        discount *= gamma

    return discounted_return_value