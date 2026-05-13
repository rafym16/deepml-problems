import numpy as np

def solve_jacobi(A, b, n):
    x = np.zeros(len(A))

    if len(A) != len(b):
        x = 'Error'

    for _ in range(n):
        x_new = x.copy()

        for i in range(len(A)):
            sum_mat = 0
            for j in range(len(b)):
                if i != j:
                    sum_mat += A[i][j] * x[j]
            x_new[i] = ((b[i] - sum_mat) / A[i][i])
        x = x_new.copy()
    return x