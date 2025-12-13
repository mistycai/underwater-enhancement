# src/metrics/uciqe.py
# Yang & Sowmya, "An Underwater Color Image Quality Evaluation Metric"

import numpy as np
import cv2
from typing import Tuple, Dict


def compute_uciqe(img_bgr: np.ndarray) -> Tuple[float, Dict]:
    c1, c2, c3 = 0.4680, 0.2745, 0.2576

    # convert to Lab and normalize
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    L = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 255.0
    b = (lab[:, :, 2] - 128.0) / 255.0

    # sigma_c: std of chroma
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = float(np.std(chroma))

    # con_l: luminance contrast (top 1% - bottom 1%)
    L_flat = L.reshape(-1)
    n = L_flat.size
    if n == 0:
        return 0.0, {"sigma_c": 0.0, "con_l": 0.0, "mu_s": 0.0}

    k = max(int(np.ceil(0.01 * n)), 1)                # number of pixels in 1%
    L_sorted = np.sort(L_flat)
    bot_1 = float(np.mean(L_sorted[:k]))
    top_1 = float(np.mean(L_sorted[-k:]))
    con_l = top_1 - bot_1

    # mu_s: mean saturation in Lab
    eps = 1e-12
    sat = chroma / np.sqrt(chroma ** 2 + L ** 2 + eps)
    mu_s = float(np.mean(sat))

    # UCIQE
    val = float(c1 * sigma_c + c2 * con_l + c3 * mu_s)

    details: Dict[str, float] = {
        "sigma_c": sigma_c,
        "con_l": con_l,
        "mu_s": mu_s,
    }
    return val, details