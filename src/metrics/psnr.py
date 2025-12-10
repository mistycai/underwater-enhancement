import cv2
import numpy as np

def compute_psnr(img1, img2):
    """
    Compute PSNR between two images (uint8, same size).
    Higher PSNR means less difference.

    Parameters:
    -----------
    img1 : ground truth image
    img2 : denoised image

    Returns:
    --------
    PSNR value (float)
    """
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # Images are identical

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
