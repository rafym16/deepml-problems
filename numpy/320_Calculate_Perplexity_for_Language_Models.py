import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.

    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]

    Returns:
        Perplexity value as a float
    """
    neg_log_likelihood = -np.log(probabilities)
    avg_loss = np.mean(neg_log_likelihood)
    token_perplexity = np.exp(avg_loss)
    return token_perplexity