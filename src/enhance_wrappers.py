# enhancement wrappers for underwater image fusion pipeline
# all functions accept bgr images and return bgr images

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

# ======================== rcc / wrcc ========================
from color_correction.rcc_wrcc import rcc_rgb, wrcc_rgb

def apply_rcc_bgr(
    bgr_img: np.ndarray,
    use_wrcc: bool = False,
    alpha: float = 0.5,
    window_size: int = 9,
    guided_radius: int = 2,
) -> np.ndarray:
    # apply rcc or wrcc color correction
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    if use_wrcc:
        rgb_out = wrcc_rgb(rgb, alpha=alpha, window_size=window_size, guided_radius=guided_radius)
    else:
        rgb_out = rcc_rgb(rgb, alpha=alpha)
    return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)


# ======================== clahe ========================
from contrast.pipeline import CLAHEConfig, CLAHEContrastEnhancer

def apply_clahe_bgr(
    bgr_img: np.ndarray,
    clip_limit: float = 6.0,
    tile_grid_size: Tuple[int, int] = (32, 32),
    use_gating: bool = False,
    sigma_threshold: float = 0.18,
    ag_threshold: float = 0.06,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # apply clahe contrast enhancement on L channel
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
        'sigma_L_out': result.sigma_L_out if result.applied else result.sigma_L,
        'contrast_gain': (result.sigma_L_out / result.sigma_L) if result.applied and result.sigma_L > 0 else 1.0,
    }
    
    return result.image, info


# ======================== denoising ========================
@dataclass
class DenoiseConfig:
    gaussian_ksize: int = 3
    gaussian_sigma: float = 1.5
    median_ksize: int = 3
    bilateral_d: int = 3
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: Optional[float] = None
    sharpen_strength: float = 0.1


def gaussian_smooth(img: np.ndarray, ksize: int = 3, sigma: float = 1.5) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img: np.ndarray, d: int = 3, sigma_color: float = 25.0,
                     sigma_space: Optional[float] = None) -> np.ndarray:
    if d is None or d <= 0:
        return img
    if sigma_space is None:
        sigma_space = max(10, d * 2)
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def laplacian_highboost(img: np.ndarray, strength: float = 0.1) -> np.ndarray:
    # sharpen using laplacian
    if strength <= 0:
        return img
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, ddepth=cv2.CV_32F, ksize=3)
    out = f - strength * lap
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise_pipeline(bgr_img: np.ndarray, config: Optional[DenoiseConfig] = None) -> np.ndarray:
    # run full denoising pipeline: gaussian -> median -> bilateral -> sharpen
    if config is None:
        config = DenoiseConfig()
    img = bgr_img.copy()
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
    bilateral_d: int = 3,
    sharpen_strength: float = 0.1,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    # apply denoising with specified parameters
    config = DenoiseConfig(
        gaussian_ksize=gaussian_ksize,
        gaussian_sigma=gaussian_sigma,
        median_ksize=median_ksize,
        bilateral_d=bilateral_d,
        sharpen_strength=sharpen_strength,
    )
    result = denoise_pipeline(bgr_img, config)
    info = {
        'gaussian_ksize': gaussian_ksize,
        'median_ksize': median_ksize,
        'bilateral_d': bilateral_d,
        'sharpen_strength': sharpen_strength,
    }
    return result, info