import torch

def calculate_matrix_mean(matrix, mode: str):
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    # Your implementation here
    if mode == "row":
        return a_t.mean(dim=0)
    elif mode == "column":
        return a_t.mean(dim=1)
    else:
        raise ValueError("Mode must be either 'row' or 'column'.")