import torch


def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.

    Args:
        matrix: A square matrix (n x n) as a torch.Tensor

    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    determinant = torch.det(matrix)
    trace = torch.trace(matrix)
    return determinant, trace

