import torch
from typing import Tuple

def shuffle_data(X, y, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly shuffle X and y together, maintaining the correspondence between samples.
    """
    if seed is not None:
        torch.manual_seed(seed)

    indices = torch.randperm(X.shape[0])
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    return (X_shuffled, y_shuffled)