import numpy as np
import cv2
from typing import Tuple, Dict


def _alpha_trimmed_mean_std(data: np.ndarray, alpha: float = 0.1) -> Tuple[float, float]:
    sorted_data = np.sort(data.flatten())
    n = len(sorted_data)
    trim = int(alpha * n / 2)
    
    if trim == 0:
        trimmed = sorted_data
    else:
        trimmed = sorted_data[trim:-trim]
    
    return np.mean(trimmed), np.std(trimmed)


def uicm(img_bgr: np.ndarray, alpha: float = 0.1) -> float:
    img = img_bgr.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    RG = R - G
    YB = (R + G) / 2.0 - B
    
    mu_rg, sigma_rg = _alpha_trimmed_mean_std(RG, alpha)
    mu_yb, sigma_yb = _alpha_trimmed_mean_std(YB, alpha)
    
    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + \
            0.1586 * np.sqrt(sigma_rg**2 + sigma_yb**2)


def _eme(channel: np.ndarray, block_size: int = 8) -> float:
    h, w = channel.shape
    k1 = h // block_size
    k2 = w // block_size
    
    if k1 == 0 or k2 == 0:
        return 0.0
    
    eme_sum = 0.0
    
    for i in range(k1):
        for j in range(k2):
            block = channel[i*block_size:(i+1)*block_size, 
                           j*block_size:(j+1)*block_size]
            i_max = block.max()
            i_min = block.min()

            if i_min > 0 and i_max > i_min:
                eme_sum += np.log(i_max / i_min)
    
    return (2.0 / (k1 * k2)) * eme_sum


def uism(img_bgr: np.ndarray, block_size: int = 8) -> float:
    img = img_bgr.astype(np.float64)

    weights = [0.114, 0.587, 0.299]  
    
    uism_val = 0.0
    
    for c in range(3):
        channel = img[:, :, c]

        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        eme_val = _eme(edge_mag, block_size)
        uism_val += weights[c] * eme_val
    
    return uism_val


def _plip_add(x: np.ndarray, y: np.ndarray, k: float = 1026.0) -> np.ndarray:
    return x + y - (x * y) / k


def _plip_sub(x: np.ndarray, y: np.ndarray, k: float = 1026.0) -> np.ndarray:
    return (x - y) / (1.0 - y / k + 1e-10)


def _plip_mult(c: float, x: np.ndarray, k: float = 1026.0) -> np.ndarray:
    return k - k * np.power(1.0 - x / k + 1e-10, c)


def _logamee(channel: np.ndarray, block_size: int = 8, k: float = 1026.0) -> float:
    h, w = channel.shape
    k1 = h // block_size
    k2 = w // block_size
    
    if k1 == 0 or k2 == 0:
        return 0.0
    
    logamee_sum = 0.0
    
    for i in range(k1):
        for j in range(k2):
            block = channel[i*block_size:(i+1)*block_size, 
                           j*block_size:(j+1)*block_size]
            i_max = block.max()
            i_min = block.min()
            
            if i_max > i_min:
                numer = _plip_sub(np.array([i_max]), np.array([i_min]), k)[0]
                denom = _plip_add(np.array([i_max]), np.array([i_min]), k)[0]
                
                if denom > 0 and numer > 0:
                    contrast = numer / denom
                    if contrast > 0:
                        logamee_sum += contrast * np.log(contrast)
    
    return (1.0 / (k1 * k2)) * logamee_sum


def uiconm(img_bgr: np.ndarray, block_size: int = 8) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return _logamee(gray, block_size)


def compute_uiqm(img_bgr: np.ndarray, block_size: int = 8) -> Tuple[float, Dict]:
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    
    _uicm = uicm(img_bgr)
    _uism = uism(img_bgr, block_size)
    _uiconm = uiconm(img_bgr, block_size)
    
    val = c1 * _uicm + c2 * _uism + c3 * _uiconm
    
    return val, {'UICM': _uicm, 'UISM': _uism, 'UIConM': _uiconm}