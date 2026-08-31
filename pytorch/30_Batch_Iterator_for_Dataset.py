import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple

def batch_iterator(X, y, seed=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use TensorDataset and DataLoader to randomly shuffle X and y together,
    maintaining the correspondence between samples.
    """
    # Hint: Use TensorDataset and DataLoader with shuffle=True.
    if seed is not None:
        torch.manual_seed(seed)

    tensor_X = torch.from_numpy(X)
    tensor_y = torch.from_numpy(y)

    dataset = TensorDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=len(X), shuffle=True)

    X_shuffled, y_shuffled = next(iter(dataloader))
    return X_shuffled, y_shuffled