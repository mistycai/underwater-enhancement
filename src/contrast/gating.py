# src/contrast/gating.py
from .stats import luminance_std, average_gradient


def should_apply_clahe(L_norm, sigma_thresh=0.10, grad_thresh=0.02):
    '''
    Decide whether to apply CLAHE
    '''
    sigma_L = luminance_std(L_norm)
    ag_L = average_gradient(L_norm)

    # low contrast / gradients -> enhance
    if sigma_L < sigma_thresh or ag_L < grad_thresh:
        return True
    else:
        return False