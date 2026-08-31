import torch


def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate the recall metric for binary classification.

    Args:
        y_true: Tensor of true binary labels (0 or 1)
        y_pred: Tensor of predicted binary labels (0 or 1)

    Returns:
        Recall value as a float
    """
    # Your implementation here
    true_positives = torch.sum((y_pred == 1) & (y_true == 1)).item()
    false_negatives = torch.sum((y_pred == 0) & (y_true == 1)).item()

    result = true_positives / (true_positives + false_negatives + 1e-7)
    return result