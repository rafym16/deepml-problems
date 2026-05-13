import numpy as np


def to_categorical(x, n_col=None):
    # Your code here
    result = []
    if n_col == None:
        n_col = x.max() + 1

    base_encoder = np.zeros(n_col)

    for item in x:
        encoder = base_encoder.copy()
        encoder[item] = 1
        result.append(encoder)

    return result