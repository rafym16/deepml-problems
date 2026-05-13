import torch


def linear_regression_gradient_descent(X, y, alpha, iterations):
    """
    Solve linear regression via gradient descent using PyTorch autograd.
    X: Tensor or convertible shape (m,n); y: shape (m,) or (m,1).
    alpha: learning rate; iterations: number of steps.
    Returns a 1-D tensor of length n, rounded to 4 decimals.
    """
    X_t = torch.as_tensor(X, dtype=torch.float)
    y_t = torch.as_tensor(y, dtype=torch.float).reshape(-1, 1)
    m, n = X_t.shape
    theta = torch.zeros((n, 1), requires_grad=True)
    # Your implementation here
    for _ in range(iterations):
        h = torch.matmul(X_t, theta)
        loss = torch.mul(1 / m, torch.sum(torch.sub(h, y_t) ** 2))

        loss.backward()

        with torch.no_grad():
            theta -= alpha * theta.grad

        theta.grad.zero_()

    return torch.round(theta.detach().flatten() * 1e4) / 1e4