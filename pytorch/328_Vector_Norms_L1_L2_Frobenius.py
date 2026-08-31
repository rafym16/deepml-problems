import torch


def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    Args:
        arr: Input tensor (1D or 2D)
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

    result = torch.linalg.norm(arr, ord=norm_map[norm_type])
    return result.item()

