import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    true_pred = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            true_pred += 1

    score = true_pred / len(y_true)
    return score