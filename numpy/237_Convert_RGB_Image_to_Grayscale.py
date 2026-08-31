import numpy as np


def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.

    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]

    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    try:
        np_image = np.array(image)
    except Exception as e:
        return -1

    if np_image.ndim != 3 or np_image.shape[2] != 3:
        return -1

    if np_image.shape[0] == 0 or np_image.shape[1] == 0:
        return -1

    if np.any(np_image < 0) or np.any(np_image > 255):
        return -1

    weights = np.array([0.2989, 0.5870, 0.1140])

    mat_mul = np_image @ weights

    grayscale = np.round(mat_mul).astype(int)

    return grayscale.tolist()