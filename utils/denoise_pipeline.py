from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from utils.quality_metrics import contrast_gain, psnr_rgb, ssim_rgb, uciqe, uiqm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineConfig:
    gaussian_ksize: int = 5
    gaussian_sigma: float = 1.5
    median_ksize: int = 3
    bilateral_d: int = 7  # set <=0 to skip
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: Optional[float] = None
    sharpen_strength: float = 0.3  # set <=0 to skip
    remove_spots: bool = False
    spot_thresh: int = 200
    spot_k: int = 5
    add_gaussian_noise_std: float = 0.0  # set >0 to add synthetic noise
    add_salt_pepper_amount: float = 0.0  # set >0 to add synthetic noise
    add_salt_vs_pepper: float = 0.5
    per_channel_sp: bool = False
    seed: Optional[int] = None  


def gaussian_smooth(img: np.ndarray, ksize: int = 5, sigma: float = 1.5) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def median_filter(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    ksize = int(ksize) | 1
    return cv2.medianBlur(img, ksize)


def bilateral_filter_optional(
    img: np.ndarray, d: int = 0, sigma_color: float = 25.0, sigma_space: Optional[float] = None
) -> np.ndarray:
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


def remove_big_spots(img_bgr: np.ndarray, thresh: int = 220, k: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.inpaint(img_bgr, mask_clean, 3, cv2.INPAINT_TELEA)


def add_gaussian_noise(
    img: np.ndarray, mean: float = 0.0, std: float = 10.0, clip_min: int = 0, clip_max: int = 255, seed: Optional[int] = None
) -> np.ndarray:
    if std <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=img.shape).astype(np.float32)
    f = img.astype(np.float32) + noise
    f = np.clip(f, clip_min, clip_max)
    return f.astype(img.dtype, copy=False)


def add_salt_pepper_noise(
    img: np.ndarray,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
    per_channel: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    if amount <= 0:
        return img.copy()
    amount = float(np.clip(amount, 0.0, 1.0))
    salt_vs_pepper = float(np.clip(salt_vs_pepper, 0.0, 1.0))
    h, w = img.shape[:2]
    total_pixels = h * w
    n_salt = int(round(total_pixels * amount * salt_vs_pepper))
    n_pepper = int(round(total_pixels * amount * (1.0 - salt_vs_pepper)))
    rng = np.random.default_rng(seed)
    noisy = img.copy()

    def _apply(coords, value, channel=None):
        if img.ndim == 2:
            noisy[coords[0], coords[1]] = value
        elif channel is None:
            noisy[coords[0], coords[1], :] = value
        else:
            noisy[coords[0], coords[1], channel] = value

    if per_channel and img.ndim == 3:
        for c in range(img.shape[2]):
            salt_coords = (
                rng.integers(0, h, n_salt, endpoint=False),
                rng.integers(0, w, n_salt, endpoint=False),
            )
            pepper_coords = (
                rng.integers(0, h, n_pepper, endpoint=False),
                rng.integers(0, w, n_pepper, endpoint=False),
            )
            _apply(salt_coords, 255, channel=c)
            _apply(pepper_coords, 0, channel=c)
    else:
        salt_coords = (
            rng.integers(0, h, n_salt, endpoint=False),
            rng.integers(0, w, n_salt, endpoint=False),
        )
        pepper_coords = (
            rng.integers(0, h, n_pepper, endpoint=False),
            rng.integers(0, w, n_pepper, endpoint=False),
        )
        _apply(salt_coords, 255)
        _apply(pepper_coords, 0)

    return noisy


def run_pipeline(img_bgr: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    out = img_bgr

    if cfg.add_gaussian_noise_std > 0:
        out = add_gaussian_noise(out, std=cfg.add_gaussian_noise_std, seed=cfg.seed)

    if cfg.add_salt_pepper_amount > 0:
        out = add_salt_pepper_noise(
            out,
            amount=cfg.add_salt_pepper_amount,
            salt_vs_pepper=cfg.add_salt_vs_pepper,
            per_channel=cfg.per_channel_sp,
            seed=cfg.seed,
        )

    if cfg.remove_spots:
        out = remove_big_spots(out, thresh=cfg.spot_thresh, k=cfg.spot_k)

    out = gaussian_smooth(out, ksize=cfg.gaussian_ksize, sigma=cfg.gaussian_sigma)
    out = median_filter(out, ksize=cfg.median_ksize)
    out = bilateral_filter_optional(
        out, d=cfg.bilateral_d, sigma_color=cfg.bilateral_sigma_color, sigma_space=cfg.bilateral_sigma_space
    )
    out = laplacian_highboost(out, strength=cfg.sharpen_strength)
    return out


def compute_metrics(orig_bgr: np.ndarray, enhanced_bgr: np.ndarray) -> Dict[str, float]:
    rgb_orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    rgb_enh = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    uiqm_orig = float(uiqm(rgb_orig))
    uiqm_enh = float(uiqm(rgb_enh))
    uciqe_orig = float(uciqe(rgb_orig))
    uciqe_enh = float(uciqe(rgb_enh))

    def pct_change(orig_val: float, new_val: float) -> float:
        if abs(orig_val) < 1e-8 or not np.isfinite(orig_val):
            return 0.0
        return float((new_val - orig_val) / abs(orig_val) * 100.0)

    psnr_db = float(psnr_rgb(rgb_orig, rgb_enh))
    ssim_val = float(ssim_rgb(rgb_orig, rgb_enh))

    contrast_ratio = float(contrast_gain(rgb_orig, rgb_enh)) 
    contrast_pct = (contrast_ratio - 1.0) * 100.0

    return {
        "uiqm_pct_change": pct_change(uiqm_orig, uiqm_enh),
        "uciqe_pct_change": pct_change(uciqe_orig, uciqe_enh),
        "contrast_pct_change": contrast_pct,
        "psnr_db": psnr_db,
        "ssim": ssim_val,
        "ssim_pct_change": (ssim_val - 1.0) * 100.0,
        "psnr_pct_change": (psnr_db / 60.0) * 100.0,
    }


def _iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTS:
            yield path


def process_image(
    img_path: Path,
    output_dir: Optional[Path],
    cfg: PipelineConfig,
    compute_metrics_flag: bool = True,
    save_output: bool = True,
) -> Dict[str, float]:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    enhanced = run_pipeline(img, cfg)

    out_path: Optional[Path] = None
    if save_output:
        if output_dir is None:
            raise ValueError("output_dir must be provided when save_output=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / (img_path.stem + "_denoised.png")
        cv2.imwrite(str(out_path), enhanced)

    metrics: Dict[str, float] = {}
    if compute_metrics_flag:
        metrics = compute_metrics(img, enhanced)

    if out_path is not None:
        metrics["output_path"] = str(out_path)
    metrics["input_path"] = str(img_path)
    return metrics


def process_directory(
    input_dir: Path,
    output_dir: Optional[Path],
    cfg: PipelineConfig,
    compute_metrics_flag: bool = True,
    save_output: bool = True,
    max_images: Optional[int] = None,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    processed = 0
    # Only apply a limit when max_images is a positive integer.
    limit = max_images if (max_images is not None and max_images > 0) else None
    print(f"Processing directory: {input_dir} Max images: {limit}")

    for img_path in _iter_images(input_dir):
        if limit is not None and processed >= limit:
            break
        processed += 1  # count attempted images to ensure we stop promptly
        try:
            res = process_image(
                img_path=img_path,
                output_dir=output_dir,
                cfg=cfg,
                compute_metrics_flag=compute_metrics_flag,
                save_output=save_output,
            )
            results.append(res)
            if save_output:
                print(f"[OK] {img_path} -> {res.get('output_path')}")
            else:
                print(f"[OK] {img_path} (metrics only)")
        except Exception as exc:
            print(f"[FAIL] {img_path}: {exc}")
    return results


def summarize_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {}
    keys = [
        "uiqm_pct_change",
        "uciqe_pct_change",
        "contrast_pct_change",
        "psnr_pct_change",
        "ssim_pct_change",
    ]
    summary: Dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
    # Also expose absolute mean PSNR/SSIM for reference.
    psnr_vals = [r["psnr_db"] for r in results if "psnr_db" in r]
    if psnr_vals:
        summary["mean_psnr_db"] = float(np.mean(psnr_vals))
    ssim_vals = [r["ssim"] for r in results if "ssim" in r]
    if ssim_vals:
        summary["mean_ssim"] = float(np.mean(ssim_vals))
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Batch denoising pipeline with metrics.")
    parser.add_argument("--input_dir", required=True, type=Path, help="Directory of images to process.")
    parser.add_argument("--output_dir", type=Path, help="Where to write denoised images. Required unless --no_save.")
    parser.add_argument("--config_seed", type=int, default=None, help="Seed for synthetic noise (if enabled).")
    parser.add_argument("--no_metrics", action="store_true", help="Skip metric computation.")
    parser.add_argument("--no_save", action="store_true", help="Do not save denoised images, compute metrics only.")
    parser.add_argument("--max_images", type=int, default=None, help="Process at most this many images (e.g., 100).")
    args = parser.parse_args()

    cfg = PipelineConfig(seed=args.config_seed)
    if not args.no_save and args.output_dir is None:
        parser.error("--output_dir is required unless --no_save is set.")

    results = process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        compute_metrics_flag=not args.no_metrics,
        save_output=not args.no_save,
        max_images=args.max_images,
    )
    summary = summarize_metrics(results)
    print(json.dumps({"summary": summary, "count": len(results)}, indent=2))

