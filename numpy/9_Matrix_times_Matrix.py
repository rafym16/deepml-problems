import numpy as np

def matrixmul(a,b):
    mat_a = np.array(a)
    mat_b = np.array(b)

    shape_a = mat_a.shape[1]
    shape_b = mat_b.shape[1]

    c = -1
    if shape_a == shape_b:
        c = np.matmul(mat_a, mat_b)

    return c