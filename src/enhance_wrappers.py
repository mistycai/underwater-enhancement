'''
Enhancement wrappers for underwater image fusion pipeline.
All functions accept BGR images and return BGR images.
'''

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


########################## RCC / WRCC ##########################
from color_correction.rcc_wrcc import rcc_rgb, wrcc_rgb
def apply_rcc_bgr(
    bgr_img: np.ndarray,
    use_wrcc: bool = True,
    alpha: float = 0.5,
    window_size: int = 9,
    guided_radius: int = 2,
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    if use_wrcc:
        rgb_out = wrcc_rgb(rgb, alpha=alpha, window_size=window_size, guided_radius=guided_radius)
    else:
        rgb_out = rcc_rgb(rgb, alpha=alpha)
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


########################## RCC --> White Balance ##########################
from color_correction.white_balance import gray_world_white_balance
def rcc_plus_white_balance(
    bgr_img: np.ndarray,
    rcc_alpha: float = 0.5,
    wb_lambda: float = 0.2,
    use_wrcc: bool = True,
    wrcc_window_size: int = 9,
    wrcc_guided_radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    rcc_result = apply_rcc_bgr(bgr_img, use_wrcc=use_wrcc, alpha=rcc_alpha,
                                window_size=wrcc_window_size, guided_radius=wrcc_guided_radius)
    final_result = gray_world_white_balance(rcc_result, lambda_param=wb_lambda)
    return final_result, rcc_result


def apply_color_correction_bgr(
    bgr_img: np.ndarray,
    method: str = 'rcc_wb',
    wb_lambda: float = 0.2,
    rcc_alpha: float = 0.5,
    use_wrcc: bool = True,
    wrcc_window_size: int = 9,
    wrcc_guided_radius: int = 2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    info = {'method': method}
    rcc_intermediate = None
    
    if method == 'gray_world':
        result = gray_world_white_balance(bgr_img, lambda_param=wb_lambda)
        info['wb_lambda'] = wb_lambda
        
    elif method == 'rcc':
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        rgb_out = rcc_rgb(rgb, alpha=rcc_alpha)
        result = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
        info['rcc_alpha'] = rcc_alpha
        info['rcc_type'] = 'basic'
        
    elif method == 'wrcc':
        result = apply_rcc_bgr(bgr_img, use_wrcc=True, alpha=rcc_alpha,
                               window_size=wrcc_window_size, guided_radius=wrcc_guided_radius)
        info['rcc_alpha'] = rcc_alpha
        info['rcc_type'] = 'wrcc'
        
    elif method == 'rcc_wb':
        result, rcc_intermediate = rcc_plus_white_balance(
            bgr_img, rcc_alpha=rcc_alpha, wb_lambda=wb_lambda, use_wrcc=use_wrcc,
            wrcc_window_size=wrcc_window_size, wrcc_guided_radius=wrcc_guided_radius)
        info['rcc_alpha'] = rcc_alpha
        info['wb_lambda'] = wb_lambda
        info['rcc_type'] = 'wrcc' if use_wrcc else 'basic'
    else:
        raise ValueError(f"Unknown color correction method: {method}")
    
    info['rcc_intermediate'] = rcc_intermediate
    return result, info


########################## CLAHE ##########################
from contrast.pipeline import CLAHEConfig, CLAHEContrastEnhancer
# from contrast.stats import luminance_std, average_gradient
# from contrast.color_space import bgr_to_lab_luminance

def apply_clahe_bgr(
    bgr_img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
    use_gating: bool = False,
    sigma_threshold: float = 0.18,
    ag_threshold: float = 0.06,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = bgr_img.shape[:2]
    tile_h = H // tile_grid_size[0]
    tile_w = W // tile_grid_size[1]
    tile_size = min(tile_h, tile_w)
    
    config = CLAHEConfig(
        tile_size=tile_size,
        clip_limit=clip_limit,
        sigma_thresh=sigma_threshold,
        grad_thresh=ag_threshold,
        use_gating=use_gating,
    )
    
    enhancer = CLAHEContrastEnhancer(config)
    result = enhancer.enhance(bgr_img)
    
    info = {
        'applied': result.applied,
        'sigma_L_in': result.sigma_L,
        'ag_in': result.grad_L,
        'sigma_L_out': result.sigma_L_out if result.applied else result.sigma_L,
        'ag_out': result.grad_L_out if result.applied else result.grad_L,
        'contrast_gain': (result.sigma_L_out / result.sigma_L) if result.applied and result.sigma_L > 0 else 1.0,
        'tile_size_pixels': tile_size,
        'gating_enabled': use_gating,
    }
    
    return result.image, info


########################## Denoising ##########################
@dataclass
class DenoiseConfig:
    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.5
    median_ksize: int = 3
    bilateral_d: int = 7
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: Optional[float] = None
    sharpen_strength: float = 0.3
    remove_spots: bool = False
    spot_thresh: int = 220
    spot_kernel_size: int = 5


def gaussian_smooth(img: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img: np.ndarray, d: int = 7, sigma_color: float = 25.0,
                     sigma_space: Optional[float] = None) -> np.ndarray:
    if d is None or d <= 0:
        return img
    if sigma_space is None:
        sigma_space = max(10, d * 2)
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def laplacian_highboost(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    if strength <= 0:
        return img
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, ddepth=cv2.CV_32F, ksize=3)
    out = f - strength * lap
    return np.clip(out, 0, 255).astype(np.uint8)


def remove_big_spots(img_bgr: np.ndarray, thresh: int = 220, kernel_size: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.inpaint(img_bgr, mask_clean, 3, cv2.INPAINT_TELEA)


def denoise_pipeline(bgr_img: np.ndarray, config: Optional[DenoiseConfig] = None) -> np.ndarray:
    if config is None:
        config = DenoiseConfig()
    img = bgr_img.copy()
    if config.remove_spots:
        img = remove_big_spots(img, config.spot_thresh, config.spot_kernel_size)
    img = gaussian_smooth(img, config.gaussian_ksize, config.gaussian_sigma)
    img = median_filter(img, config.median_ksize)
    img = bilateral_filter(img, d=config.bilateral_d, sigma_color=config.bilateral_sigma_color,
                          sigma_space=config.bilateral_sigma_space)
    img = laplacian_highboost(img, config.sharpen_strength)
    return img


def apply_denoise_bgr(
    bgr_img: np.ndarray,
    gaussian_ksize: int = 3,
    gaussian_sigma: float = 1.5,
    median_ksize: int = 3,
    bilateral_d: int = 5,
    sharpen_strength: float = 0.3,
) -> np.ndarray:
    config = DenoiseConfig(
        gaussian_ksize=gaussian_ksize, gaussian_sigma=gaussian_sigma,
        median_ksize=median_ksize, bilateral_d=bilateral_d, sharpen_strength=sharpen_strength)
    return denoise_pipeline(bgr_img, config)


def apply_denoise_pipeline(bgr_img: np.ndarray) -> np.ndarray:
    return apply_denoise_bgr(bgr_img)