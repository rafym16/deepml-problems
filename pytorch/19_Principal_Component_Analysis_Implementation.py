import torch

def pca(data, k):
    """
    Perform PCA on `data`, returning the top `k` principal components as a tensor.
    Input: Tensor or convertible of shape (n_samples, n_features).
    Returns: a torch.Tensor of shape (n_features, k), with floats rounded to 4 decimals.
    Note: If an eigenvector's first non-zero value is negative, flip its sign.
    """
    # Your implementation here
    data_t = torch.as_tensor(data, dtype=torch.float)
    mean = torch.mean(data_t, dim=0, keepdim=True)
    std = torch.std(data_t, dim=0, keepdim=True)
    data_centered = data_t - mean
    standardized_data = data_centered / std

    covariance = torch.matmul(standardized_data.T, standardized_data) / (data_t.shape[0] - 1)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    sorted_idx = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_idx]
    principal_components = eigenvectors[:, :k]

    for i in range(k):
        vector = principal_components[:, i]
        for value in vector:
            if value != 0:
                if value < 0:
                    principal_components[:, i] *= -1
                break
    return torch.round(principal_components * 1e4) / 1e4
    pass