import numpy as np

def matrix_dot_vector(a, b):
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.

	result = 0
	result_array = []

	if len(a) == len(b):
		for item in a:
			if len(item) == len(b):
				result += np.dot(item, b)
				result_array.append(result)
				result = 0
	else:
		result_array = -1
	return result_array