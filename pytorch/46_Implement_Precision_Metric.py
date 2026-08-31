import torch

def precision(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Calculates the precision metric for binary classification.

    Precision is defined as the ratio of true positives to the sum of
    true positives and false positives.

    Args:
        y_true: True binary labels (1D tensor)
        y_pred: Predicted binary labels (1D tensor)

    Returns:
        Precision value as a scalar tensor
    """
    # Your implementation here
    true_positives = torch.sum((y_pred == 1) & (y_true == 1)).item()
    false_positives = torch.sum((y_pred == 1) & (y_true == 0)).item()

    result = true_positives / (true_positives + false_positives + 1e-7)
    return result