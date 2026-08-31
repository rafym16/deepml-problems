import numpy as np

def precision(y_true, y_pred):
	# Your code here
    true_pos_value = 0
    true_neg_value = 0
    false_pos_value = 0
    false_neg_value = 0

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            true_pos_value += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            false_pos_value += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            false_neg_value += 1
        else:
            true_neg_value += 1

    precision_score = true_pos_value / (true_pos_value + false_neg_value)
    return precision_score