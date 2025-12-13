"""
Multi-scale fusion module for underwater image enhancement.

This module provides the complete Ancuti-style fusion implementation
with proper weight computation (local contrast, saliency, etc.)

Reference: Ancuti et al. "Enhancing Underwater Images and Videos by Fusion" (CVPR 2012)
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


# ============================================================================
# Utility Functions
# ============================================================================

def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1] range."""
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def resize_to_smallest(images: List[np.ndarray]) -> List[np.ndarray]:
    """Resize all images to the smallest HxW among them."""
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    H_min, W_min = min(heights), min(widths)
    
    return [cv2.resize(im, (W_min, H_min), interpolation=cv2.INTER_AREA) for im in images]


def normalize_weight(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize weight map to [0, 1] range."""
    w_min, w_max = w.min(), w.max()
    if w_max - w_min < eps:
        return np.ones_like(w)
    return (w - w_min) / (w_max - w_min)


# ============================================================================
# Weight Maps (Ancuti et al.)
# ============================================================================

def compute_laplacian_contrast_weight(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian contrast weight (W_L).
    
    Magnitude of Laplacian on grayscale image.
    Assigns high values to edges and texture.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return np.abs(lap)


def compute_local_contrast_weight(gray: np.ndarray) -> np.ndarray:
    """
    Local contrast weight (W_LC) from Ancuti et al.
    
    Computes the deviation between pixel luminance and local average
    using a 5x5 binomial kernel (1/16 * [1,4,6,4,1]).
    
    This strengthens transitions in highlighted and shadowed regions.
    """
    # 5x5 binomial kernel as specified in paper
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    
    # Low-pass filter using separable convolution
    low_pass = cv2.sepFilter2D(gray, cv2.CV_32F, kernel_1d, kernel_1d)
    
    # Local contrast = absolute difference from local average
    return np.abs(gray - low_pass)


def compute_saliency_weight(bgr: np.ndarray) -> np.ndarray:
    """
    Saliency weight (W_S) based on Achanta et al.
    
    Emphasizes discriminating objects that lose prominence underwater.
    Uses frequency-tuned center-surround contrast in Lab color space.
    """
    # Convert to Lab for perceptual uniformity
    bgr_uint8 = (bgr * 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Mean color of entire image
    mean_lab = np.mean(lab, axis=(0, 1))
    
    # Gaussian blur approximates the low-frequency component
    blurred = cv2.GaussianBlur(lab, (5, 5), 0)
    
    # Saliency = distance from mean in Lab space
    diff = blurred - mean_lab
    saliency = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return saliency


def compute_exposedness_weight(bgr: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    """
    Exposedness weight (W_E) from Ancuti et al.
    
    Evaluates how well exposed each pixel is.
    Pixels near mid-gray (0.5) get higher weights.
    
    W_E = exp(-(I - 0.5)^2 / (2*sigma^2))
    Applied per channel and multiplied.
    """
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    
    gauss_b = np.exp(-((B - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_g = np.exp(-((G - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_r = np.exp(-((R - 0.5) ** 2) / (2 * sigma ** 2))
    
    return gauss_b * gauss_g * gauss_r


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:
    """
    Saturation weight - per-pixel std across color channels.
    
    Note: This is NOT in the original Ancuti paper, but can help
    for underwater images where color restoration is important.
    """
    return np.std(bgr, axis=2)


def compute_weights_ancuti(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
) -> np.ndarray:
    """
    Compute Ancuti-style fusion weights.
    
    Combined weight is product of individual weights raised to lambda powers:
    W = W_L^λ_L * W_LC^λ_LC * W_S^λ_S * W_E^λ_E
    
    NO branch bias - weights are computed purely from image content.
    
    Args:
        bgr_img: Input BGR image
        lam_*: Exponent for each weight component
        exposedness_sigma: Sigma for exposedness Gaussian
    
    Returns:
        Combined weight map (unnormalized)
    """
    eps = 1e-12
    bgr = to_float01(bgr_img)
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Compute individual weight maps
    w_lap = normalize_weight(compute_laplacian_contrast_weight(gray))
    w_lc = normalize_weight(compute_local_contrast_weight(gray))
    w_sal = normalize_weight(compute_saliency_weight(bgr))
    w_exp = normalize_weight(compute_exposedness_weight(bgr, sigma=exposedness_sigma))
    
    # Combined weight
    W_hat = (
        (w_lap ** lam_laplacian) *
        (w_lc ** lam_local_contrast) *
        (w_sal ** lam_saliency) *
        (w_exp ** lam_exposedness)
    )
    
    return W_hat + eps


def normalize_weights(weight_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize weights across inputs so they sum to 1 at each pixel.
    
    W_k_normalized = W_k / sum(W_j for all j)
    """
    W_stack = np.stack(weight_list, axis=0)  # (N, H, W)
    W_sum = np.sum(W_stack, axis=0, keepdims=True)
    W_sum[W_sum == 0] = 1.0
    W_norm = W_stack / W_sum
    return [W_norm[i] for i in range(W_norm.shape[0])]


# ============================================================================
# Pyramids
# ============================================================================

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """Build Gaussian pyramid with specified number of levels."""
    g_pyr = [img]
    current = img
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        g_pyr.append(current)
    return g_pyr


def build_laplacian_pyramid(img: np.ndarray, levels: int) -> List[np.ndarray]:
    """
    Build Laplacian pyramid.
    
    Each level contains the detail (high frequency) information.
    Last level is the low-frequency residual.
    """
    g_pyr = build_gaussian_pyramid(img, levels)
    l_pyr = []
    
    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=size)
        lap = g_pyr[i] - up
        l_pyr.append(lap)
    
    # Last level is the low-frequency residual
    l_pyr.append(g_pyr[-1])
    return l_pyr


def collapse_laplacian_pyramid(l_pyr: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from Laplacian pyramid."""
    current = l_pyr[-1]
    
    for level in range(len(l_pyr) - 2, -1, -1):
        size = (l_pyr[level].shape[1], l_pyr[level].shape[0])
        up = cv2.pyrUp(current, dstsize=size)
        current = up + l_pyr[level]
    
    return current


# ============================================================================
# Main Fusion Functions
# ============================================================================

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
    verbose: bool = True,
) -> np.ndarray:
    """
    Multi-scale fusion of three input images (Ancuti-style).
    
    Extended from the original 2-input paper to support three branches
    (e.g., RCC, CLAHE, denoise). No branch bias - weights determined
    purely by image quality metrics.
    
    Args:
        img1_bgr: First input (e.g., RCC result)
        img2_bgr: Second input (e.g., CLAHE result)
        img3_bgr: Third input (e.g., denoised result)
        levels: Number of pyramid levels (paper uses 5)
        lam_*: Lambda exponents for weight components
        exposedness_sigma: Sigma for exposedness weight
        verbose: Print weight statistics
    
    Returns:
        Fused image (BGR, uint8)
    """
    # Align spatial sizes
    img1_bgr, img2_bgr, img3_bgr = resize_to_smallest([img1_bgr, img2_bgr, img3_bgr])
    
    # Convert to float [0, 1]
    f1 = to_float01(img1_bgr)
    f2 = to_float01(img2_bgr)
    f3 = to_float01(img3_bgr)
    
    # Compute Ancuti-style weights (NO branch bias)
    weight_params = dict(
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
        exposedness_sigma=exposedness_sigma,
    )
    
    W1_hat = compute_weights_ancuti(f1, **weight_params)
    W2_hat = compute_weights_ancuti(f2, **weight_params)
    W3_hat = compute_weights_ancuti(f3, **weight_params)
    
    # Normalize weights to sum to 1
    W1, W2, W3 = normalize_weights([W1_hat, W2_hat, W3_hat])
    
    if verbose:
        print(f"   [Fusion weights] Input1: {W1.mean():.3f}, Input2: {W2.mean():.3f}, Input3: {W3.mean():.3f}")
    
    # Build Laplacian pyramids for inputs
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    L3 = build_laplacian_pyramid(f3, levels)
    
    # Build Gaussian pyramids for weights
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    G3 = build_gaussian_pyramid(W3, levels)
    
    # Fuse at each pyramid level
    # R^l = sum_k G^l{W_k} * L^l{I_k}
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        w3_k = G3[k][:, :, np.newaxis]
        
        fused_k = w1_k * L1[k] + w2_k * L2[k] + w3_k * L3[k]
        fused_pyr.append(fused_k)
    
    # Collapse pyramid to get final image
    fused_float = collapse_laplacian_pyramid(fused_pyr)
    fused_float = np.clip(fused_float, 0.0, 1.0)
    
    return (fused_float * 255.0).astype(np.uint8)


def multi_scale_fusion_two(
    input1_bgr: np.ndarray,
    input2_bgr: np.ndarray,
    levels: int = 5,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Two-input fusion as in the original Ancuti paper.
    
    Args:
        input1_bgr: First input (color corrected)
        input2_bgr: Second input (contrast enhanced)
        levels: Number of pyramid levels
        lam_*: Lambda exponents
        verbose: Print weight statistics
    
    Returns:
        Fused image (BGR, uint8)
    """
    input1_bgr, input2_bgr = resize_to_smallest([input1_bgr, input2_bgr])
    
    f1 = to_float01(input1_bgr)
    f2 = to_float01(input2_bgr)
    
    weight_params = dict(
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
    )
    
    W1_hat = compute_weights_ancuti(f1, **weight_params)
    W2_hat = compute_weights_ancuti(f2, **weight_params)
    
    W1, W2 = normalize_weights([W1_hat, W2_hat])
    
    if verbose:
        print(f"   [Fusion weights] Input1: {W1.mean():.3f}, Input2: {W2.mean():.3f}")
    
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        fused_pyr.append(w1_k * L1[k] + w2_k * L2[k])
    
    fused_float = collapse_laplacian_pyramid(fused_pyr)
    fused_float = np.clip(fused_float, 0.0, 1.0)
    
    return (fused_float * 255.0).astype(np.uint8)


# ============================================================================
# Legacy Function (with branch bias) for Comparison
# ============================================================================

def multi_scale_fusion_three_legacy(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    img3_bgr: np.ndarray,
    levels: int = 4,
    lam_contrast: float = 1.0,
    lam_saturation: float = 1.0,
    lam_exposed: float = 1.0,
) -> np.ndarray:
    """
    Legacy fusion with simplified weights (for comparison only).
    
    Uses contrast, saturation, exposedness (not local contrast or saliency).
    """
    img1_bgr, img2_bgr, img3_bgr = resize_to_smallest([img1_bgr, img2_bgr, img3_bgr])
    
    f1 = to_float01(img1_bgr)
    f2 = to_float01(img2_bgr)
    f3 = to_float01(img3_bgr)
    
    def compute_weights_simple(bgr_img):
        eps = 1e-12
        bgr = to_float01(bgr_img)
        gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        w_c = normalize_weight(compute_laplacian_contrast_weight(gray))
        w_s = normalize_weight(compute_saturation_weight(bgr))
        w_e = normalize_weight(compute_exposedness_weight(bgr))
        
        W_hat = (w_c ** lam_contrast) * (w_s ** lam_saturation) * (w_e ** lam_exposed)
        return W_hat + eps
    
    W1_hat = compute_weights_simple(f1)
    W2_hat = compute_weights_simple(f2)
    W3_hat = compute_weights_simple(f3)
    
    W1, W2, W3 = normalize_weights([W1_hat, W2_hat, W3_hat])
    
    L1, L2, L3 = [build_laplacian_pyramid(f, levels) for f in [f1, f2, f3]]
    G1, G2, G3 = [build_gaussian_pyramid(w, levels) for w in [W1, W2, W3]]
    
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        w3_k = G3[k][:, :, np.newaxis]
        fused_pyr.append(w1_k * L1[k] + w2_k * L2[k] + w3_k * L3[k])
    
    fused_float = collapse_laplacian_pyramid(fused_pyr)
    fused_float = np.clip(fused_float, 0.0, 1.0)
    return (fused_float * 255.0).astype(np.uint8)


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_weights(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Get individual weight maps for visualization.
    
    Returns dictionary with normalized weight maps.
    """
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