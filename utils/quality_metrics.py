from typing import Tuple
import cv2
import numpy as np


def to_float_rgb(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.5:  # likely [0,255]
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def from_float_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert float32 RGB in [0,1] to uint8 [0,255].
    """
    img = np.clip(img * 255.0, 0.0, 255.0)
    return img.astype(np.uint8)


def uciqe(image_rgb: np.ndarray) -> float:
    """
    UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
    """
    img = to_float_rgb(image_rgb)
    img_uint8 = from_float_rgb(img)

    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)  # scale to [0,100]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    con_l = float(np.max(L) - np.min(L))

    s = chroma / (L + 1e-6)
    mu_s = float(np.mean(s))

    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    return float(c1 * sigma_c + c2 * con_l + c3 * mu_s)

def _uicm(image_rgb: np.ndarray) -> float:
    img = to_float_rgb(image_rgb)
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    rg = R - G
    yb = 0.5 * (R + G) - B

    mu_rg = float(np.mean(rg))
    mu_yb = float(np.mean(yb))
    sigma_rg = float(np.std(rg))
    sigma_yb = float(np.std(yb))

    uicm_val = (-0.0268 * mu_rg +
                 0.1586 * sigma_rg -
                 0.0240 * mu_yb +
                 0.1767 * sigma_yb)
    return float(uicm_val)


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def _uism(image_rgb: np.ndarray) -> float:
    img = to_float_rgb(image_rgb)
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Er = float(np.mean(_gradient_magnitude(R)))
    Eg = float(np.mean(_gradient_magnitude(G)))
    Eb = float(np.mean(_gradient_magnitude(B)))

    # Standard luminance weights
    w_r, w_g, w_b = 0.299, 0.587, 0.114
    uism_val = w_r * Er + w_g * Eg + w_b * Eb
    return float(uism_val)


def _uiconm(image_rgb: np.ndarray) -> float:
    img = to_float_rgb(image_rgb)
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return float(np.std(Y))


def uiqm(image_rgb: np.ndarray) -> float:
    uicm_val = _uicm(image_rgb)
    uism_val = _uism(image_rgb)
    uiconm_val = _uiconm(image_rgb)

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return float(c1 * uicm_val + c2 * uism_val + c3 * uiconm_val)

def luminance_contrast(image_rgb: np.ndarray) -> float:
    img = to_float_rgb(image_rgb)
    Y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return float(np.std(Y))


def contrast_gain(orig_rgb: np.ndarray, enhanced_rgb: np.ndarray) -> float:
    """
    contrast_gain = contrast(enhanced) / contrast(original)
    """
    c_orig = luminance_contrast(orig_rgb)
    c_enh = luminance_contrast(enhanced_rgb)
    if c_orig < 1e-6:
        return 0.0
    return float(c_enh / c_orig)
