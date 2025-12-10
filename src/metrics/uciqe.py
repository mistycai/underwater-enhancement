# src/metrics/uciqe.py
# ref: Yang & Sowmya, "An Underwater Color Image Quality Evaluation Metric"

import numpy as np
import cv2
from typing import Tuple, Dict


def compute_uciqe(img_bgr: np.ndarray) -> Tuple[float, Dict]:
    '''
    Underwater Colour Image Quality Evaluation 
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    
    where:
        sigma_c = std of chroma (in CIELab)
        con_l = contrast of luminance (top 1% - bottom 1%)
        mu_s = mean saturation (in HSV)
    
    Coefficients for underwater monitoring images:
        c1=0.4680, c2=0.2745, c3=0.2576    
    '''
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    
    # sigma_c
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float64)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0  # center at 0
    b = lab[:, :, 2] - 128.0  # center at 0
    
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    
    # con_l
    L_sorted = np.sort(L.flatten())
    n = len(L_sorted)
    k = max(int(np.ceil(0.01 * n)), 1)
    
    top_1_percent = L_sorted[-k:]
    bot_1_percent = L_sorted[:k]
    con_l = np.mean(top_1_percent) - np.mean(bot_1_percent)
    
    # mu_s
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    mu_s = np.mean(hsv[:, :, 1])
    
    # UCIQE
    val = c1 * sigma_c + c2 * con_l + c3 * mu_s
    
    return val, {'sigma_c': sigma_c, 'con_l': con_l, 'mu_s': mu_s}