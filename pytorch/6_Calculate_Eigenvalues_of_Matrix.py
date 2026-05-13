import torch

def calculate_eigenvalues(matrix):
    """
    Compute eigenvalues of a 2Ã2 matrix using PyTorch.
    Input: 2Ã2 tensor; Output: 1-D tensor with the two eigenvalues in ascending order.
    """
    # Your implementation here
    input_matrix = torch.tensor(matrix, dtype=torch.float32)
    eigenvalues = torch.linalg.eigvals(input_matrix)
    real_eigenvalues = torch.real(eigenvalues)
    eigenvalues, _ = torch.sort(real_eigenvalues)

    return eigenvalues
