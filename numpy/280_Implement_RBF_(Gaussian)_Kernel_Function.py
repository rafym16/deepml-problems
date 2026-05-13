import numpy as np

def rbf_kernel(X1, X2, gamma):
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.

    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)

    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Your code here
    from math import e
    result = []
    for i in range(len(X1)):
        res_rbf = []
        for j in range(len(X2)):
            diff = X1[i] - X2[j]
            distance = np.linalg.norm(diff)
            rbf = e ** (-1 * gamma * (distance ** 2))
            res_rbf.append(rbf)
        result.append(res_rbf)

    return result