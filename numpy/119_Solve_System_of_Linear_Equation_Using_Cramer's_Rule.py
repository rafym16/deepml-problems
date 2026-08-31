import numpy as np

def cramer_rule(A, b):
    array_A = np.array(A)
    array_b = np.array(b)

    n = len(b)
    x = np.zeros(n)

    det_A = np.linalg.det(array_A)

    if np.isclose(det_A, 0):
        return -1

    for i in range(n):
        A_i = array_A.copy()
        A_i[:, i] = array_b
        x[i] = np.linalg.det(A_i) / det_A

    return x


