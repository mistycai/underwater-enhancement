# multi-scale fusion for underwater image enhancement
# reference: ancuti et al. "enhancing underwater images and videos by fusion" (cvpr 2012)

import cv2
import numpy as np
from typing import List, Tuple, Dict

def to_float01(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def resize_to_smallest(images: List[np.ndarray]) -> List[np.ndarray]:
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    H_min, W_min = min(heights), min(widths)
    return [cv2.resize(im, (W_min, H_min), interpolation=cv2.INTER_AREA) for im in images]


def normalize_weight(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w_min, w_max = w.min(), w.max()
    if w_max - w_min < eps:
        return np.ones_like(w)
    return (w - w_min) / (w_max - w_min)


# ======================== weight maps ========================

def compute_laplacian_contrast_weight(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    # laplacian contrast - high values at edges/texture
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return np.abs(lap)


def compute_local_contrast_weight(gray: np.ndarray) -> np.ndarray:
    # local contrast using 5x5 binomial kernel
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    low_pass = cv2.sepFilter2D(gray, cv2.CV_32F, kernel_1d, kernel_1d)
    return np.abs(gray - low_pass)


def compute_saliency_weight(bgr: np.ndarray) -> np.ndarray:
    # saliency based on achanta et al.
    bgr_uint8 = (bgr * 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
    mean_lab = np.mean(lab, axis=(0, 1))
    blurred = cv2.GaussianBlur(lab, (5, 5), 0)
    diff = blurred - mean_lab
    return np.sqrt(np.sum(diff ** 2, axis=2))


def compute_exposedness_weight(bgr: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    # well-exposedness - pixels near mid-gray get higher weight
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    gauss_b = np.exp(-((B - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_g = np.exp(-((G - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_r = np.exp(-((R - 0.5) ** 2) / (2 * sigma ** 2))
    return gauss_b * gauss_g * gauss_r


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:
    # saturation - std across color channels
    return np.std(bgr, axis=2)


def compute_weights_ancuti(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
) -> np.ndarray:
    # compute ancuti-style fusion weights
    eps = 1e-12
    bgr = to_float01(bgr_img)
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    w_lap = normalize_weight(compute_laplacian_contrast_weight(gray))
    w_lc = normalize_weight(compute_local_contrast_weight(gray))
    w_sal = normalize_weight(compute_saliency_weight(bgr))
    w_exp = normalize_weight(compute_exposedness_weight(bgr, sigma=exposedness_sigma))
    
    W_hat = (
        (w_lap ** lam_laplacian) *
        (w_lc ** lam_local_contrast) *
        (w_sal ** lam_saliency) *
        (w_exp ** lam_exposedness)
    )
    
    return W_hat + eps


def normalize_weights(weight_list: List[np.ndarray]) -> List[np.ndarray]:
    # normalize weights to sum to 1 at each pixel
    W_stack = np.stack(weight_list, axis=0)
    W_sum = np.sum(W_stack, axis=0, keepdims=True)
    W_sum[W_sum == 0] = 1.0
    W_norm = W_stack / W_sum
    return [W_norm[i] for i in range(W_norm.shape[0])]


# ======================== pyramids ========================

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    g_pyr = [img]
    current = img
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        g_pyr.append(current)
    return g_pyr


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    g_pyr = build_gaussian_pyramid(img, levels)
    l_pyr = []
    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=size)
        l_pyr.append(g_pyr[i] - up)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def collapse_laplacian_pyramid(l_pyr: List[np.ndarray]) -> np.ndarray:
    current = l_pyr[-1]
    for level in range(len(l_pyr) - 2, -1, -1):
        size = (l_pyr[level].shape[1], l_pyr[level].shape[0])
        up = cv2.pyrUp(current, dstsize=size)
        current = up + l_pyr[level]
    return current


# ======================== main fusion ========================

def multi_scale_fusion_three(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    img3_bgr: np.ndarray,
    levels: int = 5,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
    # branch scaling factors (multiply computed weight by this factor)
    scale_input1: float = 1.0,  # rcc
    scale_input2: float = 1.0,  # clahe
    scale_input3: float = 1.0,  # denoise
    verbose: bool = True,
) -> np.ndarray:
    # 3-input multi-scale fusion with optional branch scaling
    # scale < 1.0 decreases that branch's contribution
    # scale > 1.0 increases that branch's contribution
    
    img1_bgr, img2_bgr, img3_bgr = resize_to_smallest([img1_bgr, img2_bgr, img3_bgr])
    
    f1 = to_float01(img1_bgr)
    f2 = to_float01(img2_bgr)
    f3 = to_float01(img3_bgr)
    
    weight_params = dict(
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
        exposedness_sigma=exposedness_sigma,
    )
    
    # compute content-based weights
    W1_hat = compute_weights_ancuti(f1, **weight_params)
    W2_hat = compute_weights_ancuti(f2, **weight_params)
    W3_hat = compute_weights_ancuti(f3, **weight_params)
    
    # apply branch scaling
    W1_hat = W1_hat * scale_input1
    W2_hat = W2_hat * scale_input2
    W3_hat = W3_hat * scale_input3
    
    # normalize to sum to 1
    W1, W2, W3 = normalize_weights([W1_hat, W2_hat, W3_hat])
    
    if verbose:
        print(f"   [fusion weights] rcc: {W1.mean():.3f}, clahe: {W2.mean():.3f}, denoise: {W3.mean():.3f}")
    
    # build pyramids
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    L3 = build_laplacian_pyramid(f3, levels)
    
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    G3 = build_gaussian_pyramid(W3, levels)
    
    # fuse at each level
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        w3_k = G3[k][:, :, np.newaxis]
        fused_k = w1_k * L1[k] + w2_k * L2[k] + w3_k * L3[k]
        fused_pyr.append(fused_k)
    
    # collapse pyramid
    fused_float = collapse_laplacian_pyramid(fused_pyr)
    fused_float = np.clip(fused_float, 0.0, 1.0)
    
    return (fused_float * 255.0).astype(np.uint8)


# ======================== visualization ========================

def visualize_weights(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
) -> Dict[str, np.ndarray]:
    # get individual weight maps for visualization
    bgr = to_float01(bgr_img)
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    w_lap = normalize_weight(compute_laplacian_contrast_weight(gray))
    w_lc = normalize_weight(compute_local_contrast_weight(gray))
    w_sal = normalize_weight(compute_saliency_weight(bgr))
    w_exp = normalize_weight(compute_exposedness_weight(bgr))
    
    combined = (
        (w_lap ** lam_laplacian) *
        (w_lc ** lam_local_contrast) *
        (w_sal ** lam_saliency) *
        (w_exp ** lam_exposedness)
    )
    
    return {
        'laplacian_contrast': w_lap,
        'local_contrast': w_lc,
        'saliency': w_sal,
        'exposedness': w_exp,
        'combined': normalize_weight(combined),
    }