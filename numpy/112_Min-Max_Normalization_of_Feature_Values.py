import numpy as np

def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].

    Args:
        x: A list of numerical values

    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    array = np.array(x)
    denominator = array.max() - array.min()
    if denominator != 0:
        result = (array - array.min()) / denominator
        return result
    else:
        return [0.0] * len(array)
    pass