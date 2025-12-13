"""
Multi-scale Laplacian pyramid fusion for underwater image enhancement.

Implements the fusion framework from:
Ancuti et al. "Enhancing Underwater Images and Videos by Fusion" (CVPR 2012)

Extended to support three input branches (RCC, CLAHE, denoise).
"""

import numpy as np
import cv2

from .weights import to_float01, compute_weights
from .pyramids import build_gaussian_pyramid, build_laplacian_pyramid, collapse_laplacian_pyramid


def normalize_weight_maps(weight_list: list) -> list:
    """
    Normalize weight maps so they sum to 1 at each pixel.
    
    W_k_normalized = W_k / sum(W_k for all k)
    
    Args:
        weight_list: List of weight maps (one per input)
    
    Returns:
        List of normalized weight maps
    """
    W_stack = np.stack(weight_list, axis=0)  # (N, H, W)
    W_sum = np.sum(W_stack, axis=0, keepdims=True)
    W_sum[W_sum == 0] = 1.0  # Avoid division by zero
    W_norm = W_stack / W_sum
    return [W_norm[i] for i in range(W_norm.shape[0])]


def multi_scale_fusion(
    rcc: np.ndarray,
    clahe: np.ndarray,
    denoise: np.ndarray,
    levels: int = 5,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
    verbose: bool = True
) -> np.ndarray:
    """
    Multi-scale Laplacian pyramid fusion of three enhancement branches.
    
    Implements Ancuti-style fusion:
    1. Compute weight maps based on image quality metrics
    2. Build Laplacian pyramids for inputs
    3. Build Gaussian pyramids for weights
    4. Fuse at each pyramid level
    5. Collapse to get final result
    
    Args:
        rcc: Red Channel Compensation result (BGR)
        clahe: CLAHE result (BGR)
        denoise: Denoised result (BGR)
        levels: Number of pyramid levels (paper uses 5)
        lam_*: Lambda exponents for weight components
        exposedness_sigma: Sigma for exposedness weight (paper uses 0.25)
        verbose: Print weight statistics
    
    Returns:
        Fused image (BGR, uint8)
    """
    # Resize all inputs to smallest common size
    h = min(img.shape[0] for img in [rcc, clahe, denoise])
    w = min(img.shape[1] for img in [rcc, clahe, denoise])
    rcc = cv2.resize(rcc, (w, h), interpolation=cv2.INTER_AREA)
    clahe = cv2.resize(clahe, (w, h), interpolation=cv2.INTER_AREA)
    denoise = cv2.resize(denoise, (w, h), interpolation=cv2.INTER_AREA)
    
    # Convert to float [0, 1]
    f1 = to_float01(rcc)
    f2 = to_float01(clahe)
    f3 = to_float01(denoise)
    
    # Compute Ancuti-style weights (no branch bias)
    weight_params = dict(
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
        exposedness_sigma=exposedness_sigma,
    )
    
    W1_hat = compute_weights(f1, 'rcc', **weight_params)
    W2_hat = compute_weights(f2, 'clahe', **weight_params)
    W3_hat = compute_weights(f3, 'denoise', **weight_params)
    
    # Normalize weights to sum to 1
    W1, W2, W3 = normalize_weight_maps([W1_hat, W2_hat, W3_hat])
    
    if verbose:
        print(f"   [Fusion weights] RCC: {W1.mean():.3f}, CLAHE: {W2.mean():.3f}, Denoise: {W3.mean():.3f}")
    
    # Build Laplacian pyramids for inputs
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    L3 = build_laplacian_pyramid(f3, levels)
    
    # Build Gaussian pyramids for normalized weights
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    G3 = build_gaussian_pyramid(W3, levels)
    
    # Fuse at each pyramid level
    # R^l(x,y) = sum_k G^l{W_k}(x,y) * L^l{I_k}(x,y)
    fused_pyr = []
    for level in range(levels):
        # Expand weight dimensions for broadcasting with color channels
        w1 = G1[level][:, :, np.newaxis]
        w2 = G2[level][:, :, np.newaxis]
        w3 = G3[level][:, :, np.newaxis]
        
        # Weighted sum at this level
        fused_level = w1 * L1[level] + w2 * L2[level] + w3 * L3[level]
        fused_pyr.append(fused_level)
    
    # Collapse Laplacian pyramid to get final result
    fused = collapse_laplacian_pyramid(fused_pyr)
    
    # Clip and convert to uint8
    fused = np.clip(fused, 0.0, 1.0)
    return (fused * 255).astype(np.uint8)


def multi_scale_fusion_two_input(
    input1: np.ndarray,
    input2: np.ndarray,
    levels: int = 5,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    verbose: bool = True
) -> np.ndarray:
    """
    Two-input fusion as in the original Ancuti paper.
    
    The paper uses:
    - Input 1: White-balanced (color corrected) image
    - Input 2: Contrast-enhanced (CLAHE on denoised) image
    
    Args:
        input1: First input (e.g., white-balanced)
        input2: Second input (e.g., contrast-enhanced)
        levels: Number of pyramid levels
        lam_*: Lambda exponents for weight components
        verbose: Print weight statistics
    
    Returns:
        Fused image (BGR, uint8)
    """
    # Resize to common size
    h = min(input1.shape[0], input2.shape[0])
    w = min(input1.shape[1], input2.shape[1])
    input1 = cv2.resize(input1, (w, h), interpolation=cv2.INTER_AREA)
    input2 = cv2.resize(input2, (w, h), interpolation=cv2.INTER_AREA)
    
    # Convert to float
    f1 = to_float01(input1)
    f2 = to_float01(input2)
    
    # Compute weights
    weight_params = dict(
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
    )
    
    W1_hat = compute_weights(f1, **weight_params)
    W2_hat = compute_weights(f2, **weight_params)
    
    W1, W2 = normalize_weight_maps([W1_hat, W2_hat])
    
    if verbose:
        print(f"   [Fusion weights] Input1: {W1.mean():.3f}, Input2: {W2.mean():.3f}")
    
    # Build pyramids
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    
    # Fuse
    fused_pyr = []
    for level in range(levels):
        w1 = G1[level][:, :, np.newaxis]
        w2 = G2[level][:, :, np.newaxis]
        fused_pyr.append(w1 * L1[level] + w2 * L2[level])
    
    fused = collapse_laplacian_pyramid(fused_pyr)
    fused = np.clip(fused, 0.0, 1.0)
    return (fused * 255).astype(np.uint8)


def get_weight_visualization(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
) -> dict:
    """
    Get individual weight maps for visualization/debugging.
    
    Args:
        bgr_img: Input BGR image
        lam_*: Lambda exponents
    
    Returns:
        Dictionary with individual and combined weight maps
    """
    from .weights import (
        compute_laplacian_contrast_weight,
        compute_local_contrast_weight,
        compute_saliency_weight,
        compute_exposedness_weight,
        normalize_weight,
    )
    
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
        'combined': combined,
    }