import torch

def gaussian_elimination(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
        Solves the system Ax = b using Gaussian Elimination with partial pivoting.

        :param A: Coefficient matrix (torch.Tensor)
        :param b: Right-hand side vector (torch.Tensor)
        :return: Solution vector x (torch.Tensor)
        """
    n = len(b)

    A = A.clone()
    b = b.clone()

    x = torch.zeros(n, dtype=A.dtype, device=A.device)

    # 1. Forward Elimination dengan Partial Pivoting
    for k in range(n - 1):

        # Cari baris dengan nilai absolut terbesar
        max_row = k + torch.argmax(
            torch.abs(A[k:n, k])
        ).item()

        # Cek apakah pivot = 0
        if torch.isclose(
            A[max_row, k],
            torch.zeros_like(A[max_row, k])
        ).item():
            raise ValueError(
                "Matriks singular/ill-conditioned! "
                "Tidak memiliki solusi unik."
            )

        # Tukar baris
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # Eliminasi
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]

            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # 2. Back Substitution
    x = torch.zeros(n, dtype=A.dtype, device=A.device)

    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (
            b[i] -
            torch.dot(A[i, i + 1:], x[i + 1:])
        ) / A[i, i]

    return x