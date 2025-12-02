# src/fusion/multiscale_fusion.py
'''
Fixes:
1. Reduced well-exposedness weight (was causing denoise to dominate)
2. Added branch-specific bias to favor better-performing branches
3. Increased contrast weight to favor CLAHE output
4. Added saliency-based weighting option
'''

import cv2
import numpy as np

######## helpers ########
def to_float01(img: np.ndarray) -> np.ndarray:
    '''
    Convert image to float32 in [0, 1]
    Accepts uint8 [0,255] or float arrays (either [0,1] or [0,255]).
    '''
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f = f / 255.0
    return np.clip(f, 0.0, 1.0)


def resize_to_smallest(images):
    '''Resize all images to the smallest HxW'''
    heights = [im.shape[0] for im in images]
    widths = [im.shape[1] for im in images]
    H_min = min(heights)
    W_min = min(widths)

    resized = []
    for im in images:
        resized.append(cv2.resize(im, (W_min, H_min), interpolation=cv2.INTER_AREA))
    return resized



######## weight maps (Ancuti et al.) ########


def compute_contrast_weight(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    '''
    Magnitude of Laplacian on grayscale.
    Higher values = more edge detail = better for fusion
    '''
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    w = np.abs(lap)
    return w


def compute_saturation_weight(bgr: np.ndarray) -> np.ndarray:
    '''
    Per-pixel standard deviation across color channels.
    Higher saturation = more colorful = better color restoration
    '''
    return np.std(bgr, axis=2)


def compute_well_exposedness_weight(bgr: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    ''''
    Product of Gaussians centered at 0.5 for each channel.
    TODO: it favors images closest to mid-gray (0.5), so the denoise branch (~original) dominates.
    Use a larger sigma to make it less selective
    '''
    B = bgr[:, :, 0]
    G = bgr[:, :, 1]
    R = bgr[:, :, 2]

    gauss_b = np.exp(-0.5 * ((B - 0.5) ** 2) / (sigma ** 2))
    gauss_g = np.exp(-0.5 * ((G - 0.5) ** 2) / (sigma ** 2))
    gauss_r = np.exp(-0.5 * ((R - 0.5) ** 2) / (sigma ** 2))

    w = gauss_b * gauss_g * gauss_r
    return w


def compute_sharpness_weight(gray: np.ndarray) -> np.ndarray:
    '''
    Sharpness weight based on gradient magnitude
    '''
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    return gradient_mag


def compute_colorfulness_weight(bgr: np.ndarray) -> np.ndarray:
    '''
    Favors regions with good color variety
    '''
    R, G, B = bgr[:, :, 2], bgr[:, :, 1], bgr[:, :, 0]
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    
    # local colorfulness using a window
    kernel = np.ones((11, 11), np.float32) / 121
    rg_local = cv2.filter2D(rg, -1, kernel)
    yb_local = cv2.filter2D(yb, -1, kernel)
    
    colorfulness = np.sqrt(rg_local**2 + yb_local**2)
    return colorfulness


def compute_weights_improved(
    bgr_img: np.ndarray,
    branch_name: str = 'unknown',
    lam_contrast: float = 2.0,
    lam_saturation: float = 1.5,
    lam_exposed: float = 0.3,
    lam_sharpness: float = 1.0,
    use_colorfulness: bool = True,
) -> np.ndarray:
    '''
    Compute combined weight map for one branch with IMPROVED weighting
    '''
    eps = 1e-12
    bgr = to_float01(bgr_img)
    gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # individual weight components
    w_c = compute_contrast_weight(gray)
    w_s = compute_saturation_weight(bgr)
    w_e = compute_well_exposedness_weight(bgr, sigma=0.35)  # Larger sigma = less selective
    w_sh = compute_sharpness_weight(gray)
    
    # normalize
    def norm01(w):
        w_min = w.min()
        w_max = w.max()
        if w_max - w_min < eps:
            return np.ones_like(w)
        return (w - w_min) / (w_max - w_min)

    w_c = norm01(w_c)
    w_s = norm01(w_s)
    w_e = norm01(w_e)
    w_sh = norm01(w_sh)

    # combined weight with adjusted powers
    W_hat = (w_c ** lam_contrast) * (w_s ** lam_saturation) * (w_e ** lam_exposed) * (w_sh ** lam_sharpness)
    
    if use_colorfulness:
        w_color = norm01(compute_colorfulness_weight(bgr))
        W_hat = W_hat * (w_color ** 0.5)
    
    # based on empirical results: CLAHE performs best
    branch_bias = {
        'rcc': 1.2,
        'clahe': 1.5,
        'denoise': 0.6,
        'unknown': 1.0
    }
    bias = branch_bias.get(branch_name.lower(), 1.0)
    W_hat = W_hat * bias
    
    W_hat = W_hat + eps
    return W_hat


def normalize_weights(weight_list):
    '''
        W_i = W_hat_i / sum_j W_hat_j
    '''
    W_stack = np.stack(weight_list, axis=0)  # (N, H, W)
    sum_w = np.sum(W_stack, axis=0, keepdims=True)
    sum_w[sum_w == 0] = 1.0
    W_norm = W_stack / sum_w
    return [W_norm[i] for i in range(W_norm.shape[0])]


######## Pyramids ########

def build_gaussian_pyramid(img: np.ndarray, levels: int):
    g_pyr = [img]
    current = img
    for _ in range(1, levels):
        current = cv2.pyrDown(current)
        g_pyr.append(current)
    return g_pyr


def build_laplacian_pyramid(img: np.ndarray, levels: int):
    g_pyr = build_gaussian_pyramid(img, levels)
    l_pyr = []
    for i in range(levels - 1):
        size = (g_pyr[i].shape[1], g_pyr[i].shape[0])
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=size)
        lap = g_pyr[i] - up
        l_pyr.append(lap)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def collapse_laplacian_pyramid(l_pyr):
    '''Reconstruct image from Laplacian pyramid'''
    current = l_pyr[-1]
    for level in range(len(l_pyr) - 2, -1, -1):
        size = (l_pyr[level].shape[1], l_pyr[level].shape[0])
        up = cv2.pyrUp(current, dstsize=size)
        current = up + l_pyr[level]
    return current


######## Main fusion ########
def multi_scale_fusion_three(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    img3_bgr: np.ndarray,
    levels: int = 4,
    lam_contrast: float = 2.0,
    lam_saturation: float = 1.5,
    lam_exposed: float = 0.3,
) -> np.ndarray:

    # align spatial size
    img1_bgr, img2_bgr, img3_bgr = resize_to_smallest([img1_bgr, img2_bgr, img3_bgr])

    # convert to float [0,1]
    f1 = to_float01(img1_bgr)
    f2 = to_float01(img2_bgr)
    f3 = to_float01(img3_bgr)

    # compute weights per branch (with bias)
    W1_hat = compute_weights_improved(f1, branch_name='rcc', 
                                       lam_contrast=lam_contrast,
                                       lam_saturation=lam_saturation, 
                                       lam_exposed=lam_exposed)
    W2_hat = compute_weights_improved(f2, branch_name='clahe',
                                       lam_contrast=lam_contrast,
                                       lam_saturation=lam_saturation,
                                       lam_exposed=lam_exposed)
    W3_hat = compute_weights_improved(f3, branch_name='denoise',
                                       lam_contrast=lam_contrast,
                                       lam_saturation=lam_saturation,
                                       lam_exposed=lam_exposed)

    W1, W2, W3 = normalize_weights([W1_hat, W2_hat, W3_hat])
    
    print(f"   [Fusion weights] RCC: {W1.mean():.3f}, CLAHE: {W2.mean():.3f}, Denoise: {W3.mean():.3f}")

    # build pyramids
    L1 = build_laplacian_pyramid(f1, levels)
    L2 = build_laplacian_pyramid(f2, levels)
    L3 = build_laplacian_pyramid(f3, levels)

    G1 = build_gaussian_pyramid(W1, levels)
    G2 = build_gaussian_pyramid(W2, levels)
    G3 = build_gaussian_pyramid(W3, levels)

    # fuse at each scale
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
    fused_uint8 = (fused_float * 255.0).astype(np.uint8)
    return fused_uint8



######## legacy wrapper for comparison ########
def multi_scale_fusion_three_legacy(
    img1_bgr: np.ndarray,
    img2_bgr: np.ndarray,
    img3_bgr: np.ndarray,
    levels: int = 4,
    lam_contrast: float = 1.0,
    lam_saturation: float = 1.0,
    lam_exposed: float = 1.0,
) -> np.ndarray:
    '''
    Original fusion implementation
    '''
    img1_bgr, img2_bgr, img3_bgr = resize_to_smallest([img1_bgr, img2_bgr, img3_bgr])
    
    f1 = to_float01(img1_bgr)
    f2 = to_float01(img2_bgr)
    f3 = to_float01(img3_bgr)
    
    # original weight
    def compute_weights_original(bgr_img):
        eps = 1e-12
        bgr = to_float01(bgr_img)
        gray = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        w_c = compute_contrast_weight(gray)
        w_s = compute_saturation_weight(bgr)
        w_e = compute_well_exposedness_weight(bgr)
        
        def norm01(w):
            w_min, w_max = w.min(), w.max()
            if w_max - w_min < eps:
                return np.ones_like(w)
            return (w - w_min) / (w_max - w_min)
        
        w_c, w_s, w_e = norm01(w_c), norm01(w_s), norm01(w_e)
        W_hat = (w_c ** lam_contrast) * (w_s ** lam_saturation) * (w_e ** lam_exposed)
        return W_hat + eps
    
    W1_hat = compute_weights_original(f1)
    W2_hat = compute_weights_original(f2)
    W3_hat = compute_weights_original(f3)
    
    W1, W2, W3 = normalize_weights([W1_hat, W2_hat, W3_hat])
    
    L1, L2, L3 = [build_laplacian_pyramid(f, levels) for f in [f1, f2, f3]]
    G1, G2, G3 = [build_gaussian_pyramid(w, levels) for w in [W1, W2, W3]]
    
    fused_pyr = []
    for k in range(levels):
        w1_k = G1[k][:, :, np.newaxis]
        w2_k = G2[k][:, :, np.newaxis]
        w3_k = G3[k][:, :, np.newaxis]
        fused_k = w1_k * L1[k] + w2_k * L2[k] + w3_k * L3[k]
        fused_pyr.append(fused_k)
    
    fused_float = collapse_laplacian_pyramid(fused_pyr)
    fused_float = np.clip(fused_float, 0.0, 1.0)
    return (fused_float * 255.0).astype(np.uint8)