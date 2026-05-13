import torch

def transform_matrix(A, T, S):
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    det_T_t = torch.linalg.det(T_t)
    det_S_t = torch.linalg.det(S_t)

    if not (det_T_t == 0 or det_S_t == 0):
        inv_T_t = torch.inverse(T_t)
        trans_T_t = torch.matmul(inv_T_t, torch.matmul(A_t, S_t))
        return trans_T_t
    else:
        return torch.as_tensor(-1., dtype=torch.float)