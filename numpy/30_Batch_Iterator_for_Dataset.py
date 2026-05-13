import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    # Your code here
    n_sample = X.shape[0]
    batches = []

    if y is not None:
        for i in range(0, n_sample, batch_size):
            batches.append([X[i:i + batch_size], y[i:i + batch_size]])
    else:
        for i in range(0, n_sample, batch_size):
            batches.append([X[i:i + batch_size]])

    return batches