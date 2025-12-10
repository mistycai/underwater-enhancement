import numpy as np
import cv2
from typing import List


def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Gaussian pyramid with specified number of levels."""
    pyr = [img]
    for _ in range(1, levels):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Laplacian pyramid with specified number of levels."""
    g_pyr = build_gaussian_pyramid(img, levels)
    l_pyr = []
    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=size)
        l_pyr.append(g_pyr[i] - up)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def collapse_laplacian_pyramid(l_pyr: List[np.ndarray]) -> np.ndarray:
    """Collapse Laplacian pyramid to reconstruct image."""
    current = l_pyr[-1]
    for level in range(len(l_pyr) - 2, -1, -1):
        size = (l_pyr[level].shape[1], l_pyr[level].shape[0])
        current = cv2.pyrUp(current, dstsize=size) + l_pyr[level]
    return current