import numpy as np
def scalar_multiply(matrix, scalar):
    result = []
    matrix_array = np.array(matrix)
    for item in matrix_array:
        multiplication = item * scalar
        result.append(multiplication)
    return result