def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    from collections import Counter
    data = Counter(samples)
    list_result = []
    if samples:
        for item, count in data.items():
            prob = count / len(samples)
            list_result.append((item, prob))

    return list_result