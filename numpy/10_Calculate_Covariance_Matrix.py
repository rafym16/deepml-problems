import numpy as np
def calculate_covariance_matrix(vectors):
	# Your code here
	if vectors:
		cov = np.cov(vectors, dtype=float)
		return cov
	return []