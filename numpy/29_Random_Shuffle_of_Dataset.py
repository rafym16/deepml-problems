import numpy as np

def shuffle_data(X, y, seed=None):
    # Your code here
    np.random.seed(seed)

    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)

    X_shuffled = data[:, :2]
    y_shuffled = data[:, 2]

    result = (X_shuffled, y_shuffled)
    return result