import numpy as np

def cosine_similarity(v1, v2):
	# Implement your code here
    if (v1.shape == v2.shape):
        dot_prod = np.dot(v1, v2)
        mat_mul = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosine_similarity = dot_prod / mat_mul
        return np.round(cosine_similarity, 3)
	pass