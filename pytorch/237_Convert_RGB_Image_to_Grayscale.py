import torch

def rgb_to_grayscale(image):
    """
        Convert an RGB image to grayscale using luminosity method.

        Args:
            image: RGB image as list or torch.Tensor of shape (H, W, 3)
                   with values in range [0, 255]

        Returns:
            Grayscale image as 2D list with integer values,
            or -1 if input is invalid
        """

    try:
        if not isinstance(image, torch.Tensor):
            tensor_image = torch.tensor(image, dtype=torch.float32)
        else:
            tensor_image = image.float()
    except Exception as e:
        return torch.tensor(-1).item()

    if tensor_image.ndim != 3 or tensor_image.shape[2] != 3:
        return torch.tensor(-1).item()

    if tensor_image.shape[0] == 0 or tensor_image.shape[1] == 0:
        return torch.tensor(-1).item()

    if torch.any(tensor_image < 0) or torch.any(tensor_image > 255):
        return torch.tensor(-1).item()

    weights = torch.tensor([0.2989, 0.5870, 0.1140])

    mat_mul = torch.matmul(tensor_image,weights)

    grayscale = torch.round(mat_mul).to(torch.int32)

    return grayscale.tolist()