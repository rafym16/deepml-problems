import torch


def inverse_2x2(matrix):
    """
    Compute the inverse of a 2x2 matrix using PyTorch.

    Args:
        matrix: A 2x2 matrix (can be list, numpy array, or torch.Tensor)

    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here
    det_m = torch.linalg.det(m)
    if det_m != 0:
        inverse = torch.linalg.inv(m)
        return inverse
    else:
        return None
