def transform_basis(B, C):
	import numpy as np
	inv_c = np.linalg.inv(C)
	p = inv_c @ B
	return p