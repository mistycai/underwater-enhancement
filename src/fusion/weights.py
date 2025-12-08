import numpy as np
import cv2


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1] range."""
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def compute_contrast_weight(gray: np.ndarray) -> np.ndarray:
    """Laplacian-based contrast weight."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(lap)


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:
    """Standard deviation across color channels."""
    return np.std(bgr, axis=2)


def compute_well_exposedness_weight(bgr: np.ndarray, sigma: float = 0.35) -> np.ndarray:
    """
    Gaussian-based well-exposedness weight.
    Using larger sigma (0.35 vs 0.25) to be less selective.
    """
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    gauss_b = np.exp(-0.5 * ((B - 0.5) ** 2) / (sigma ** 2))
    gauss_g = np.exp(-0.5 * ((G - 0.5) ** 2) / (sigma ** 2))
    gauss_r = np.exp(-0.5 * ((R - 0.5) ** 2) / (sigma ** 2))
    return gauss_b * gauss_g * gauss_r


def compute_sharpness_weight(gray: np.ndarray) -> np.ndarray:
    """Sobel-based sharpness weight."""
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(sobelx**2 + sobely**2)


def compute_weights(
    bgr_img: np.ndarray,
    branch_name: str = 'unknown',
    lam_contrast: float = 2.0,
    lam_saturation: float = 1.5,
    lam_exposed: float = 0.3,
    lam_sharpness: float = 1.0
) -> np.ndarray:
    """
    Compute fusion weights with branch-specific bias.
    
    Args:
        bgr_img: Input BGR image
        branch_name: One of 'rcc', 'clahe', 'denoise', 'unknown'
        lam_*: Exponent weights for each component
    
    Returns:
        Weight map for fusion
    """
    eps = 1e-12
    bgr = to_float01(bgr_img)
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    w_c = compute_contrast_weight(gray)
    w_s = compute_saturation_weight(bgr)
    w_e = compute_well_exposedness_weight(bgr, sigma=0.35)
    w_sh = compute_sharpness_weight(gray)
    
    def norm01(w):
        w_min, w_max = w.min(), w.max()
        if w_max - w_min < eps:
            return np.ones_like(w)
        return (w - w_min) / (w_max - w_min)

    w_c, w_s, w_e, w_sh = norm01(w_c), norm01(w_s), norm01(w_e), norm01(w_sh)
    
    W_hat = (w_c ** lam_contrast) * (w_s ** lam_saturation) * (w_e ** lam_exposed) * (w_sh ** lam_sharpness)
    
    # Branch-specific bias
    branch_bias = {'rcc': 1.0, 'clahe': 1.5, 'denoise': 0.3, 'unknown': 1.0}
    W_hat = W_hat * branch_bias.get(branch_name.lower(), 1.0)
    
    return W_hat + eps