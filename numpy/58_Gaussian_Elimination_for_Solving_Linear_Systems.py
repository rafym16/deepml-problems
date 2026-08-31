import numpy as np


def gaussian_elimination(A, b):
    """Solves Ax = b using Gaussian Elimination with Partial Pivoting (Chapra 9.4)

    :param A: Coefficient matrix (n x n)
    :param b: Right-hand side vector (n)
    :return: Solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # 1. Forward Elimination dengan Partial Pivoting
    for k in range(n - 1):
        # --- SUBBAB 9.4: PARTIAL PIVOTING ---
        # Cari indeks baris dengan nilai mutlak terbesar di kolom k (dari baris k s.d. n-1)
        max_row = k + np.argmax(np.abs(A[k:n, k]))

        # Cek apakah matriks singular (pivot bernilai 0 atau sangat dekat dengan 0)
        if np.isclose(A[max_row, k], 0):
            raise ValueError(
                "Matriks singular/ill-conditioned! Tidak memiliki solusi unik."
            )

        # Tukar baris (swap rows) jika baris dengan elemen terbesar bukan baris k
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        # --- ELIMINASI BIASA ---
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]

            # Menggunakan vectorization NumPy agar lebih cepat
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # 2. Back Substitution
    x = np.zeros(n)
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x