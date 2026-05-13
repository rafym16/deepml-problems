def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)

    # Your implementation here
    x = torch.zeros(len(A_t))

    if len(A_t) != len(b_t):
        x = 'Error'

    for _ in range(n):
        x_new = x.clone()

        for i in range(len(A)):
            sum_mat = 0
            for j in range(len(b)):
                if i != j:
                    sum_mat += A[i][j] * x[j]
            x_new[i] = ((b[i] - sum_mat) / A[i][i])
        x = x_new.clone()

    return x