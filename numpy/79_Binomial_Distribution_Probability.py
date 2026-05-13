import math

def binomial_probability(n, k, p):
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.

    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial

    Returns:
        Probability of k successes
    """
    # Your code here
    result = math.comb(n, k) * (p ** k) * (1 - p) ** (n - k)
    return result