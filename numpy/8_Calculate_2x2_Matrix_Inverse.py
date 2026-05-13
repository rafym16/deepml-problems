import numpy as np

def inverse_2x2(matrix):
    inverse = None
    det_matrix = np.linalg.det(matrix)
    if det_matrix != 0:
        inverse = np.linalg.inv(matrix)
        return inverse
    return inverse