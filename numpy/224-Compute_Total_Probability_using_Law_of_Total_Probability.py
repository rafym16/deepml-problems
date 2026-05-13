def law_of_total_probability(priors: dict, conditionals: dict) -> float:
    """
    Compute P(A) using the Law of Total Probability.

    Args:
        priors: Dictionary mapping partition event names to P(Bi)
        conditionals: Dictionary mapping partition event names to P(A|Bi)

    Returns:
        float: The total probability P(A), rounded to 4 decimal places
    """
    # Your code here
    total_probability = 0
    if len(priors) == len(conditionals):
        for prior_key, prior_value in priors.items():
            for conditional_key, conditional_value in conditionals.items():
                if prior_key == conditional_key:
                    total_probability += (prior_value * conditional_value)
                else:
                    continue

    return total_probability