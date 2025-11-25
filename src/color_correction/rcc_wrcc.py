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
    img = np.clip(img * 255.0, 0.0, 255.0)
    return img.astype(np.uint8)

def guided_filter(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    ksize: Tuple[int, int] = (2 * r + 1, 2 * r + 1)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=ksize, normalize=True)
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=ksize, normalize=True)
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=ksize, normalize=True)

    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=ksize, normalize=True)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize, normalize=True)
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize, normalize=True)

    q = mean_a * I + mean_b
    return q

# RCC
def rcc_rgb(image_rgb: np.ndarray, alpha: float = 1.2, eps: float = 1e-6) -> np.ndarray:
    img = to_float_rgb(image_rgb)

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    mu_R = float(np.mean(R)) + eps
    mu_G = float(np.mean(G))

    scale = (mu_G / mu_R) ** alpha
    R_prime = np.clip(R * scale, 0.0, 1.0)

    out = np.stack([R_prime, G, B], axis=-1)
    return from_float_rgb(out)

# WRCC
def wrcc_rgb(
    image_rgb: np.ndarray,
    alpha: float = 1.2,
    window_size: int = 9,
    guided_radius: int = 8,
    guided_eps: float = 1e-3,
    eps: float = 1e-6,
) -> np.ndarray:
    img = to_float_rgb(image_rgb)

    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    ksize: Tuple[int, int] = (window_size, window_size)

    local_mean_R = cv2.boxFilter(R, ddepth=-1, ksize=ksize, normalize=True)
    local_mean_G = cv2.boxFilter(G, ddepth=-1, ksize=ksize, normalize=True)

    w_R = local_mean_R
    w_G = local_mean_G

    ratio = (w_G * G) / (w_R * R + eps)
    scale = np.power(ratio, alpha)

    R_prime = np.clip(R * scale, 0.0, 1.0)

    R_refined = guided_filter(
        I=G.astype(np.float32),
        p=R_prime.astype(np.float32),
        r=guided_radius,
        eps=guided_eps,
    )
    R_refined = np.clip(R_refined, 0.0, 1.0)

    out = np.stack([R_refined, G, B], axis=-1)
    return from_float_rgb(out)
