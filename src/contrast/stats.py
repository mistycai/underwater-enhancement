import cv2
import numpy as np


def luminance_std(L_norm: np.ndarray) -> float:
    return float(np.std(L_norm))


def average_gradient(L_norm: np.ndarray) -> float:
    L32 = L_norm.astype(np.float32)
    # sobel gradients
    grad_x = cv2.Sobel(L32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L32, cv2.CV_32F, 0, 1, ksize=3)
    # magnitude
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(np.mean(mag))