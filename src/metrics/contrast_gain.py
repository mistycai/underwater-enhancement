# src/metrics/contrast_gain.py
import cv2
import numpy as np

def contrast_gain(original: np.ndarray, enhanced: np.ndarray) -> float:
    '''Enhanced/original luminance std ration'''
    def get_std(img):
        return np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64))
    c_orig = get_std(original)
    return get_std(enhanced) / c_orig if c_orig > 0 else 1.0