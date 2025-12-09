# src/metrics/colorfulness.py
import numpy as np

def colorfulness(img_bgr: np.ndarray) -> float:
    '''Hasler & Süsstrunk colorfulness'''
    img = img_bgr.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg, yb = R - G, 0.5 * (R + G) - B
    sigma_rgyb = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mu_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    return sigma_rgyb + 0.3 * mu_rgyb