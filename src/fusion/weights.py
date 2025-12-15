import numpy as np
import cv2


def to_float01(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def compute_laplacian_contrast_weight(gray: np.ndarray) -> np.ndarray:

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(lap)


def compute_local_contrast_weight(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:

    # 5x5 binomial kernel: (1/16) * [1, 4, 6, 4, 1] separable
    if kernel_size == 5:
        kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    else:
        kernel_1d = cv2.getGaussianKernel(kernel_size, -1).flatten().astype(np.float32)
    
    low_pass = cv2.sepFilter2D(gray, cv2.CV_32F, kernel_1d, kernel_1d)
    
    w_lc = np.abs(gray - low_pass)
    
    return w_lc


def compute_saliency_weight(bgr: np.ndarray) -> np.ndarray:

    bgr_uint8 = (bgr * 255).astype(np.uint8)
    lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    mean_lab = np.mean(lab, axis=(0, 1))

    blurred = cv2.GaussianBlur(lab, (5, 5), 0)

    diff = blurred - mean_lab
    saliency = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return saliency


def compute_exposedness_weight(bgr: np.ndarray, sigma: float = 0.25) -> np.ndarray:

    B, G, R = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    
    gauss_b = np.exp(-((B - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_g = np.exp(-((G - 0.5) ** 2) / (2 * sigma ** 2))
    gauss_r = np.exp(-((R - 0.5) ** 2) / (2 * sigma ** 2))
    
    return gauss_b * gauss_g * gauss_r


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:

    return np.std(bgr, axis=2)


def normalize_weight(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:

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
    eps = 1e-12
    bgr = to_float01(bgr_img)
    
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    w_lap = compute_laplacian_contrast_weight(gray)
    w_lc = compute_local_contrast_weight(gray)
    w_sal = compute_saliency_weight(bgr)
    w_exp = compute_exposedness_weight(bgr, sigma=exposedness_sigma)
    
    w_lap = normalize_weight(w_lap)
    w_lc = normalize_weight(w_lc)
    w_sal = normalize_weight(w_sal)
    w_exp = normalize_weight(w_exp)

    W_hat = (
        (w_lap ** lam_laplacian) *
        (w_lc ** lam_local_contrast) *
        (w_sal ** lam_saliency) *
        (w_exp ** lam_exposedness)
    )
    if use_saturation:
        w_sat = normalize_weight(compute_saturation_weight(bgr))
        W_hat = W_hat * (w_sat ** lam_saturation)
    
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

    return compute_weights_ancuti(
        bgr_img,
        lam_laplacian=lam_laplacian,
        lam_local_contrast=lam_local_contrast,
        lam_saliency=lam_saliency,
        lam_exposedness=lam_exposedness,
        exposedness_sigma=exposedness_sigma,
        use_saturation=False,  
    )


def compute_weights_with_bias(
    bgr_img: np.ndarray,
    branch_name: str = 'unknown',
    lam_laplacian: float = 1.0,
    lam_local_contrast: float = 1.0,
    lam_saliency: float = 1.0,
    lam_exposedness: float = 1.0,
    branch_bias: dict = None,
) -> np.ndarray:
    
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