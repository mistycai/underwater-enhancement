"""
Ancuti-style weight computation for multi-scale fusion.

Implements four weight maps from the paper:
1. Laplacian contrast weight (W_L) - global edge/texture detection
2. Local contrast weight (W_LC) - highlight/shadow transitions  
3. Saliency weight (W_S) - emphasize objects lost in murky water
4. Exposedness weight (W_E) - favor mid-tones

Reference: Ancuti et al. "Enhancing Underwater Images and Videos by Fusion" (CVPR 2012)
"""

import numpy as np
import cv2


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1] range."""
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def compute_laplacian_contrast_weight(gray: np.ndarray) -> np.ndarray:
    """
    Laplacian contrast weight (W_L) from Ancuti et al.
    
    Applies Laplacian filter to luminance and takes absolute value.
    Assigns high values to edges and texture.
    """
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(lap)


def compute_local_contrast_weight(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Local contrast weight (W_LC) from Ancuti et al.
    
    Computes standard deviation between pixel luminance and local average.
    Uses a binomial kernel approximation as described in the paper.
    
    W_LC(x,y) = || I^k - I^k_whc ||
    
    where I^k_whc is low-pass filtered using 5x5 binomial kernel
    with high frequency cutoff w_hc = pi/2.75
    
    Args:
        gray: Grayscale image in [0, 1]
        kernel_size: Size of binomial kernel (default 5 as in paper)
    
    Returns:
        Local contrast weight map
    """
    # 5x5 binomial kernel: (1/16) * [1, 4, 6, 4, 1] separable
    # This approximates Gaussian and is computationally efficient
    if kernel_size == 5:
        # 1D binomial coefficients normalized
        kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    else:
        # Fall back to Gaussian approximation for other sizes
        kernel_1d = cv2.getGaussianKernel(kernel_size, -1).flatten().astype(np.float32)
    
    # Separable convolution for efficiency
    low_pass = cv2.sepFilter2D(gray, cv2.CV_32F, kernel_1d, kernel_1d)
    
    # Local contrast is the absolute difference from local average
    w_lc = np.abs(gray - low_pass)
    
    return w_lc


def compute_saliency_weight(bgr: np.ndarray) -> np.ndarray:
    """
    Saliency weight (W_S) based on Achanta et al.'s frequency-tuned method.
    
    This is a simplified but effective implementation of the saliency
    computation used in the Ancuti paper. It emphasizes discriminating
    objects that lose prominence in underwater scenes.
    
    The method computes center-surround contrast by comparing each pixel
    to the mean image color in a DoG (Difference of Gaussians) fashion.
    
    Args:
        bgr: BGR image in [0, 1]
    
    Returns:
        Saliency weight map
    """
    # Convert to Lab color space for perceptual uniformity
    bgr_uint8 = (bgr * 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # Compute mean color of the image
    mean_lab = np.mean(lab, axis=(0, 1))
    
    # Apply Gaussian blur to get the "surround" - approximates DoG
    # The paper uses frequency-tuned approach; we approximate with blur
    blurred = cv2.GaussianBlur(lab, (5, 5), 0)
    
    # Saliency = Euclidean distance from mean in Lab space
    # This highlights regions that differ from the average scene color
    diff = blurred - mean_lab
    saliency = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return saliency


def compute_exposedness_weight(bgr: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    """
    Exposedness weight (W_E) from Ancuti et al.
    
    Evaluates how well a pixel is exposed using Gaussian distance to 0.5.
    Pixels close to mid-gray (0.5) get higher weights.
    
    W_E(x,y) = exp(-(I^k(x,y) - 0.5)^2 / (2 * sigma^2))
    
    Applied to each channel and multiplied together.
    
    Args:
        bgr: BGR image in [0, 1]
        sigma: Standard deviation (paper uses 0.25)
    
    Returns:
        Exposedness weight map
    """
    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    
    gauss_b = np.exp(-((B - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_g = np.exp(-((G - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_r = np.exp(-((R - 0.5) ** 2) / (2 * sigma ** 2))
    
    return gauss_b * gauss_g * gauss_r


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:
    """
    Saturation weight - standard deviation across color channels.
    
    This is NOT in the original Ancuti paper but can be useful
    for underwater images where color restoration matters.
    
    Args:
        bgr: BGR image in [0, 1]
    
    Returns:
        Saturation weight map
    """
    return np.std(bgr, axis=2)


def normalize_weight(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize weight map to [0, 1] range."""
    w_min, w_max = w.min(), w.max()
    if w_max - w_min < eps:
        return np.ones_like(w)
    return (w - w_min) / (w_max - w_min)


def compute_weights_ancuti(
    bgr_img: np.ndarray,
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
    use_saturation: bool = False,
    lam_saturation: float = 1.0,
) -> np.ndarray:
    """
    Compute Ancuti-style fusion weights WITHOUT branch bias.
    
    This implements the four weight maps from the paper:
    - Laplacian contrast (W_L)
    - Local contrast (W_LC)
    - Saliency (W_S)
    - Exposedness (W_E)
    
    The combined weight is the product of individual weights raised
    to their respective lambda powers.
    
    Args:
        bgr_img: Input BGR image
        lam_*: Exponent for each weight component
        exposedness_sigma: Sigma for exposedness Gaussian (paper uses 0.25)
        use_saturation: Whether to include saturation weight (not in paper)
        lam_saturation: Exponent for saturation if used
    
    Returns:
        Combined weight map (unnormalized - will be normalized across inputs later)
    """
    eps = 1e-12
    bgr = to_float01(bgr_img)
    
    # Convert to grayscale for luminance-based weights
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Compute individual weight maps
    w_lap = compute_laplacian_contrast_weight(gray)
    w_lc = compute_local_contrast_weight(gray)
    w_sal = compute_saliency_weight(bgr)
    w_exp = compute_exposedness_weight(bgr, sigma=exposedness_sigma)
    
    # Normalize each to [0, 1]
    w_lap = normalize_weight(w_lap)
    w_lc = normalize_weight(w_lc)
    w_sal = normalize_weight(w_sal)
    w_exp = normalize_weight(w_exp)
    
    # Combined weight (product with lambda exponents)
    W_hat = (
        (w_lap ** lam_laplacian) *
        (w_lc ** lam_local_contrast) *
        (w_sal ** lam_saliency) *
        (w_exp ** lam_exposedness)
    )
    
    # Optionally include saturation (not in original paper)
    if use_saturation:
        w_sat = normalize_weight(compute_saturation_weight(bgr))
        W_hat = W_hat * (w_sat ** lam_saturation)
    
    # Add epsilon to avoid zero weights
    return W_hat + eps


def compute_weights(
    bgr_img: np.ndarray,
    branch_name: str = 'unknown',
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    exposedness_sigma: float = 0.25,
) -> np.ndarray:
    """
    Main weight computation function - Ancuti-style without branch bias.
    
    This is the recommended entry point. Branch name is accepted for
    compatibility but NOT used for bias (weights are computed purely
    from image content).
    
    Args:
        bgr_img: Input BGR image
        branch_name: Ignored (kept for API compatibility)
        lam_*: Exponent weights for each component
        exposedness_sigma: Sigma for exposedness weight
    
    Returns:
        Weight map for fusion
    """
    return compute_weights_ancuti(
        bgr_img,
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
        exposedness_sigma=exposedness_sigma,
        use_saturation=False,  # Stick to paper formulation
    )


# ============================================================================
# Legacy function for comparison - includes branch bias
# ============================================================================

def compute_weights_with_bias(
    bgr_img: np.ndarray,
    branch_name: str = 'unknown',
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    branch_bias: dict = None,
) -> np.ndarray:
    """
    Legacy weight computation WITH branch bias.
    
    Provided for comparison - not recommended for production use.
    The bias overrides natural weight computation.
    """
    if branch_bias is None:
        branch_bias = {'rcc': 1.0, 'clahe': 1.5, 'denoise': 0.3, 'unknown': 1.0}
    
    W_hat = compute_weights_ancuti(
        bgr_img,
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
    )
    
    bias = branch_bias.get(branch_name.lower(), 1.0)
    return W_hat * bias