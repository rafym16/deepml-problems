import numpy as np

def calculate_matrix_mean(matrix, mode):
    matrix = np.array(matrix)
    if mode == 'column':
        means = np.mean(matrix, axis=0)
    elif mode == 'row':
        means = np.mean(matrix, axis=1)
    else:
        means = 0

    return means