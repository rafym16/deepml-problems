import numpy as np

def reshape_matrix(a, new_shape):
    # Write your code here and return a python list after reshaping by using numpy's tolist() method
    matrix = np.array(a)
    try:
        reshaped_matrix = np.reshape(matrix, new_shape)

    except ValueError:
        reshaped_matrix = []

    return reshaped_matrix