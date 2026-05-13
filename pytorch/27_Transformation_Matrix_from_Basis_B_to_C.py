import torch

def transform_basis(B, C):
    """Return the change-of-basis matrix **P = Câ»Â¹ B**.

    - *B*, *C* may be 2Ã2 or 3Ã3 nested lists.
    - Result is rounded to 4 decimals and returned as a nested list.
    """
    # Your implementation here
    B_t = torch.as_tensor(B, dtype=torch.float)
    C_t = torch.as_tensor(C, dtype=torch.float)
    inv_C = torch.inverse(C_t)
    transformed_C = torch.matmul(inv_C, B_t)
    rounded_C = torch.round(transformed_C * 1e4) / 1e4
    return rounded_C.tolist()
