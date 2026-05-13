import numpy as np

def matrix_image(A):
    A = A.astype(float)
    m, n = A.shape

    R = A.copy()
    pivot_cols = []

    row = 0

    for col in range(n):

        pivot = None
        for r in range(row, m):
            if abs(R[r, col]) > 1e-10:
                pivot = r
                break

        if pivot is None:
            continue

        # swap
        R[[row, pivot]] = R[[pivot, row]]

        # normalize
        R[row] /= R[row, col]

        # eliminate
        for r in range(m):
            if r != row:
                R[r] -= R[r, col] * R[row]

        pivot_cols.append(col)

        row += 1

        if row == m:
            break

    return A[:, pivot_cols]

matrix = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

print(matrix_image(matrix))