import numpy as np

def pca(data: np.ndarray, k: int):
    """
    Perform PCA and return the top k principal components.

    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return

    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # Your code here
    # 1. Standardized data
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0, ddof=1)
    data_centered = data - mean
    standardized_data = (data - mean) / std

    # 2. Compute covariance
    cov = (standardized_data.T @ standardized_data) / (data.shape[0] - 1)

    # 3. Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 4. Sort eigenvectors descending
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = eigenvectors[:, :k]

    for i in range(k):
        vector = principal_components[:, i]
        for value in vector:
            if value != 0:
                if value < 0:
                    principal_components[:, i] *= -1
                break

    projected_data = np.dot(standardized_data, principal_components)
    return np.round(principal_components, 4)

    pass