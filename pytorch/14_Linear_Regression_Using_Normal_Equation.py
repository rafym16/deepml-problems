import torch

def linear_regression_normal_equation(X, y):
    """
    Solve linear regression via the normal equation using PyTorch.
    X: Tensor or convertible of shape (m,n); y: shape (m,) or (m,1).
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1, 1)
    # Your implementation here
    X_t_transpose = torch.transpose(X_t, 0, 1)
    y_t_transpose = torch.transpose(y_t, 0, 1)

    a = torch.matmul(X_t_transpose, X_t)
    b = torch.matmul(X_t_transpose, y_t)

    a_inverse = torch.inverse(a)

    result = torch.matmul(a_inverse, b)
    return result