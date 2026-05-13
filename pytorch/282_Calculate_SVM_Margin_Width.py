import torch

def svm_margin_width(w: torch.Tensor) -> float:
    """
    Calculate the margin width of a linear SVM classifier.

    Parameters:
    w : torch.Tensor - weight vector defining the hyperplane

    Returns:
    float - the total margin width
    """
    norm_w = torch.norm(w, p=2, keepdim=True, dtype=torch.float32)
    margin_width = 2 / norm_w
    return margin_width.item()