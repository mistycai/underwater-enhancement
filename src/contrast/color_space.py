# src/contrast/color_space.py
import cv2
import numpy as np


def bgr_to_lab_luminance(bgr_img: np.ndarray) -> tuple:
    lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab_img[:, :, 0].astype(np.float32)
    L_norm = L / 255.0
    
    return L_norm, lab_img


def lab_luminance_to_bgr(L_norm: np.ndarray, lab_img: np.ndarray) -> np.ndarray:
    L_uint8 = np.clip(L_norm * 255.0, 0, 255).astype(np.uint8)
    
    lab_out = lab_img.copy()
    lab_out[:, :, 0] = L_uint8
    
    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    return bgr_out