import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    norm_map = {
        'l1': 1,
        'l2': 2,
        'frobenius': 'fro'
    }

    result = np.linalg.norm(arr, ord=norm_map[norm_type])
    return result