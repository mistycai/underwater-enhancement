# src/contrast/stats.py
import cv2
import numpy as np


def luminance_std(L_norm):
    '''
    Compute global standard deviation of luminance
    '''
    return float(L_norm.std())


def average_gradient(L_norm):
    '''
    Compute average gradient magnitude using Sobel filters.
    '''
    L32 = L_norm.astype(np.float32)
    grad_x = cv2.Sobel(L32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L32, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return float(mag.mean())