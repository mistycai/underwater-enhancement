import numpy as np
from .stats import luminance_std, average_gradient


def should_apply_clahe(
    L_norm: np.ndarray,
    sigma_thresh: float = 0.18,
    grad_thresh: float = 0.06,
) -> bool:
    sigma_L = luminance_std(L_norm)
    ag_L = average_gradient(L_norm)
    return (sigma_L < sigma_thresh) or (ag_L < grad_thresh)


def compute_gating_stats(L_norm: np.ndarray) -> dict:
    return {
        "sigma_L": float(luminance_std(L_norm)),
        "grad_L": float(average_gradient(L_norm)),
    }