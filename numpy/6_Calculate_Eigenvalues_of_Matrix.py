import numpy as np

def calculate_eigenvalues(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues