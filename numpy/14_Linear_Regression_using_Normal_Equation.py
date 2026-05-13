import numpy as np
def linear_regression_normal_equation(X, y):
	# Your code here, make sure to round
    x_transpose = np.transpose(X)
    a = x_transpose @ X
    b = x_transpose @ y
    theta = np.linalg.inv(a) @ b
    return theta

