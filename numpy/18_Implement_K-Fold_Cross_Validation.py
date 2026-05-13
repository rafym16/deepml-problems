import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    # Your code here
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    fold_size = int(n_samples // k)

    if shuffle:
        np.random.shuffle(indices)

    splits = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size

        test_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]], axis=0)
        X_train = X[train_indices]
        X_test = X[test_indices]
        splits.append((X_train, X_test))

    return splits

    pass

result = k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=True)
print(result)