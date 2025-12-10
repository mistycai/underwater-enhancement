import numpy as np
import cv2

from .weights import to_float01, compute_weights
from .pyramids import build_gaussian_pyramid, build_laplacian_pyramid, collapse_laplacian_pyramid


def multi_scale_fusion(
    rcc: np.ndarray,
    clahe: np.ndarray,
    denoise: np.ndarray,
    levels: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """
    Multi-scale fusion of three enhancement branches.
    
    Args:
        rcc: Red Channel Compensation result (BGR)
        clahe: CLAHE result (BGR)
        denoise: Denoised result (BGR)
        levels: Number of pyramid levels
        verbose: Print weight statistics
    
    Returns:
        Fused image (BGR, uint8)
    """
    # Resize to smallest common size
    h = min(img.shape[0] for img in [rcc, clahe, denoise])
    w = min(img.shape[1] for img in [rcc, clahe, denoise])
    rcc = cv2.resize(rcc, (w, h))
    clahe = cv2.resize(clahe, (w, h))
    denoise = cv2.resize(denoise, (w, h))
    
    # Convert to float [0, 1]
    f1, f2, f3 = to_float01(rcc), to_float01(clahe), to_float01(denoise)
    
    # Compute weights with branch bias
    W1 = compute_weights(f1, 'rcc')
    W2 = compute_weights(f2, 'clahe')
    W3 = compute_weights(f3, 'denoise')
    
    # Normalize weights to sum to 1
    W_stack = np.stack([W1, W2, W3], axis=0)
    W_sum = np.sum(W_stack, axis=0, keepdims=True)
    W_sum[W_sum == 0] = 1.0
    W1, W2, W3 = W_stack[0] / W_sum[0], W_stack[1] / W_sum[0], W_stack[2] / W_sum[0]
    
    if verbose:
        print(f"   [Fusion weights] RCC: {W1.mean():.3f}, CLAHE: {W2.mean():.3f}, Denoise: {W3.mean():.3f}")
    
    # Build pyramids
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    L3 = build_laplacian_pyramid(f3, levels)
    
    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    G3 = build_gaussian_pyramid(W3, levels)
    
    # Fuse at each pyramid level
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        w3_k = G3[k][:, :, np.newaxis]
        fused_pyr.append(w1_k * L1[k] + w2_k * L2[k] + w3_k * L3[k])
    
    # Collapse pyramid
    fused = collapse_laplacian_pyramid(fused_pyr)
    
    return (np.clip(fused, 0, 1) * 255).astype(np.uint8)