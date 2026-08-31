import torch

def cramers_rule(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve system of linear equations Ax = b using Cramer's Rule.
    Returns solution vector x or -1 if no unique solution exists.
    """
    tensor_A = torch.tensor(A, dtype=torch.float32)
    tensor_b = torch.tensor(b, dtype=torch.float32)
    det_A = torch.linalg.det(tensor_A)

    if torch.isclose(det_A, torch.tensor(0.0)):
        return torch.tensor(-1).item()

    n = len(b)
    x = torch.zeros(n)
    for i in range(n):
        A_i = torch.clone(tensor_A)
        A_i[:, i] = tensor_b
        x[i] = torch.linalg.det(A_i) / det_A

    return x

A = [[2, -1, 3], [4, 2, 1], [-6, 1, -2]]
b = [5, 10, -3]

result = cramers_rule(A, b)
print(result)
