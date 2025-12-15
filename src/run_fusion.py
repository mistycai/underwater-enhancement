# 3-input underwater image fusion pipeline
# inputs: rcc (color), clahe (contrast), denoise (noise reduction)

import os
import sys
import cv2
import numpy as np
import argparse
import glob
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# import enhancement modules
try:
    from enhance_wrappers import (
        apply_rcc_bgr,
        apply_clahe_bgr,
        apply_denoise_bgr,
        DenoiseConfig,
    )
    from multiscale_fusion import multi_scale_fusion_three
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from enhance_wrappers import (
        apply_rcc_bgr,
        apply_clahe_bgr,
        apply_denoise_bgr,
        DenoiseConfig,
    )
    from multiscale_fusion import multi_scale_fusion_three

from metrics.compute_metrics import *


@dataclass
class FusionConfig:
    # rcc parameters (use_rcc with alpha=0.5 is best)
    use_wrcc: bool = False
    rcc_alpha: float = 0.5
    rcc_window_size: int = 9
    rcc_guided_radius: int = 2
    
    # clahe parameters (clip=6, tile=32 is best for detection)
    clahe_clip_limit: float = 6.0
    clahe_tile_grid_size: Tuple[int, int] = (32, 32)
    clahe_use_gating: bool = False
    
    # denoise parameters (3,3,3,0.1 is best)
    denoise_gaussian_ksize: int = 3
    denoise_gaussian_sigma: float = 1.5
    denoise_median_ksize: int = 3
    denoise_bilateral_d: int = 3
    denoise_sharpen_strength: float = 0.1
    
    # fusion parameters
    fusion_levels: int = 5
    lam_laplacian: float = 1.0
    lam_local_contrast: float = 1.0
    lam_saliency: float = 1.0
    lam_exposedness: float = 1.0
    exposedness_sigma: float = 0.25
    
    # branch scaling (1.0 = neutral, <1.0 = decrease weight, >1.0 = increase weight)
    scale_rcc: float = 1.0
    scale_clahe: float = 1.0
    scale_denoise: float = 0.2  # default: reduce denoise contribution


@dataclass
class FusionResult:
    fused: np.ndarray
    rcc: np.ndarray
    clahe: np.ndarray
    denoise: np.ndarray
    original: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)


def run_fusion_pipeline(
    bgr_img: np.ndarray,
    config: Optional[FusionConfig] = None,
    verbose: bool = True,
) -> FusionResult:
    # run 3-input fusion: rcc + clahe + denoise
    if config is None:
        config = FusionConfig()
    
    H, W = bgr_img.shape[:2]
    info = {'original_size': (H, W)}
    
    # input 1: rcc color correction
    if verbose:
        print("   [1/4] applying rcc...")
    bgr_rcc = apply_rcc_bgr(
        bgr_img,
        use_wrcc=config.use_wrcc,
        alpha=config.rcc_alpha,
        window_size=config.rcc_window_size,
        guided_radius=config.rcc_guided_radius,
    )
    bgr_rcc = cv2.resize(bgr_rcc, (W, H), interpolation=cv2.INTER_LINEAR)
    info['rcc_alpha'] = config.rcc_alpha
    
    # input 2: clahe contrast enhancement
    if verbose:
        print("   [2/4] applying clahe...")
    bgr_clahe, clahe_info = apply_clahe_bgr(
        bgr_img,
        clip_limit=config.clahe_clip_limit,
        tile_grid_size=config.clahe_tile_grid_size,
        use_gating=config.clahe_use_gating,
    )
    bgr_clahe = cv2.resize(bgr_clahe, (W, H), interpolation=cv2.INTER_LINEAR)
    info['clahe_gain'] = clahe_info.get('contrast_gain', 1.0)
    
    # input 3: denoising
    if verbose:
        print("   [3/4] applying denoise...")
    bgr_denoise, _ = apply_denoise_bgr(
        bgr_img,
        gaussian_ksize=config.denoise_gaussian_ksize,
        gaussian_sigma=config.denoise_gaussian_sigma,
        median_ksize=config.denoise_median_ksize,
        bilateral_d=config.denoise_bilateral_d,
        sharpen_strength=config.denoise_sharpen_strength,
    )
    bgr_denoise = cv2.resize(bgr_denoise, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # multi-scale fusion
    if verbose:
        print("   [4/4] applying fusion...")
    fused = multi_scale_fusion_three(
        bgr_rcc,
        bgr_clahe,
        bgr_denoise,
        levels=config.fusion_levels,
        lam_laplacian=config.lam_laplacian,
        lam_local_contrast=config.lam_local_contrast,
        lam_saliency=config.lam_saliency,
        lam_exposedness=config.lam_exposedness,
        exposedness_sigma=config.exposedness_sigma,
        scale_input1=config.scale_rcc,
        scale_input2=config.scale_clahe,
        scale_input3=config.scale_denoise,
        verbose=verbose,
    )
    
    return FusionResult(
        fused=fused,
        rcc=bgr_rcc,
        clahe=bgr_clahe,
        denoise=bgr_denoise,
        original=bgr_img,
        info=info,
    )


def process_single_image(
    input_path: str,
    output_dir: str,
    config: Optional[FusionConfig] = None,
    save_intermediates: bool = False,
    evaluate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    # process one image through fusion pipeline
    os.makedirs(output_dir, exist_ok=True)
    
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[error] could not read: {input_path}")
        return {'success': False, 'path': input_path}
    
    base = os.path.splitext(os.path.basename(input_path))[0]
    
    if verbose:
        print(f"\nprocessing: {input_path}")
        print(f"   size: {bgr.shape[1]}x{bgr.shape[0]}")
    
    result = run_fusion_pipeline(bgr, config, verbose=verbose)
    
    # save fused result
    fused_path = os.path.join(output_dir, f"{base}_fused.jpg")
    cv2.imwrite(fused_path, result.fused)
    if verbose:
        print(f"   saved: {fused_path}")
    
    # save intermediates
    if save_intermediates:
        cv2.imwrite(os.path.join(output_dir, f"{base}_rcc.jpg"), result.rcc)
        cv2.imwrite(os.path.join(output_dir, f"{base}_clahe.jpg"), result.clahe)
        cv2.imwrite(os.path.join(output_dir, f"{base}_denoise.jpg"), result.denoise)
    
    # evaluate metrics
    metrics_list = []
    if evaluate:
        if verbose:
            print("   computing metrics...")
        metrics_list = [
            compute_metrics(result.original, result.original, "original"),
            compute_metrics(result.rcc, result.original, "rcc"),
            compute_metrics(result.clahe, result.original, "clahe"),
            compute_metrics(result.denoise, result.original, "denoise"),
            compute_metrics(result.fused, result.original, "fused"),
        ]
        if verbose:
            print_metrics_table(metrics_list, base)
        
        csv_path = os.path.join(output_dir, f"{base}_metrics.csv")
        save_metrics_csv(metrics_list, csv_path)
    
    ret = {
        'success': True,
        'path': input_path,
        'output_path': fused_path,
        **result.info,
    }
    
    if metrics_list:
        orig_m = metrics_list[0]
        fused_m = metrics_list[-1]
        ret['uiqm_original'] = orig_m.uiqm
        ret['uiqm_fused'] = fused_m.uiqm
        ret['uiqm_delta'] = fused_m.uiqm - orig_m.uiqm
        ret['uciqe_original'] = orig_m.uciqe
        ret['uciqe_fused'] = fused_m.uciqe
        ret['uciqe_delta'] = fused_m.uciqe - orig_m.uciqe
        ret['contrast_gain'] = fused_m.contrast
    
    return ret


def collect_image_paths(samples: List[str]) -> List[str]:
    # collect image paths from files, directories, glob patterns
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    paths = []
    
    for s in samples:
        s = os.path.expanduser(s)
        if os.path.isdir(s):
            for root, _, files in os.walk(s):
                for f in files:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(root, f))
        elif any(ch in s for ch in ['*', '?', '[']):
            for p in glob.glob(s):
                if os.path.isfile(p) and p.lower().endswith(exts):
                    paths.append(p)
        elif os.path.isfile(s) and s.lower().endswith(exts):
            paths.append(s)
    
    return sorted(list(set(paths)))


def run_batch_processing(
    samples: List[str],
    output_dir: str,
    config: Optional[FusionConfig] = None,
    save_intermediates: bool = False,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    # batch process multiple images
    paths = collect_image_paths(samples)
    if not paths:
        print("[error] no valid image files found.")
        return []
    
    if max_images and max_images < len(paths):
        paths = paths[:max_images]
    
    os.makedirs(output_dir, exist_ok=True)
    fused_dir = os.path.join(output_dir, 'fused')
    os.makedirs(fused_dir, exist_ok=True)
    
    if save_intermediates:
        for subdir in ['rcc', 'clahe', 'denoise']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    print("=" * 60)
    print(f"batch processing: {len(paths)} images")
    print(f"output: {output_dir}")
    print("=" * 60)
    
    results = []
    iterator = tqdm(paths, desc="processing", unit="img") if len(paths) > 1 else paths
    
    for path in iterator:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            results.append({'success': False, 'path': path})
            continue
        
        base = os.path.splitext(os.path.basename(path))[0]
        
        try:
            result = run_fusion_pipeline(bgr, config, verbose=False)
            
            fused_path = os.path.join(fused_dir, f"{base}_fused.jpg")
            cv2.imwrite(fused_path, result.fused)
            
            if save_intermediates:
                cv2.imwrite(os.path.join(output_dir, 'rcc', f"{base}_rcc.jpg"), result.rcc)
                cv2.imwrite(os.path.join(output_dir, 'clahe', f"{base}_clahe.jpg"), result.clahe)
                cv2.imwrite(os.path.join(output_dir, 'denoise', f"{base}_denoise.jpg"), result.denoise)
            
            results.append({
                'success': True,
                'path': path,
                'output_path': fused_path,
                **result.info,
            })
        except Exception as e:
            results.append({'success': False, 'path': path, 'error': str(e)})
    
    # write report
    csv_path = os.path.join(output_dir, 'report.csv')
    fieldnames = ['path', 'success', 'rcc_alpha', 'clahe_gain']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    success_count = sum(1 for r in results if r.get('success', False))
    print("=" * 60)
    print(f"completed: {success_count}/{len(results)} images")
    print(f"report: {csv_path}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='3-input underwater image fusion')
    
    # input/output
    parser.add_argument('input', nargs='+', help='input image(s), directory, or glob pattern')
    parser.add_argument('--output', '-o', default='fusion_output', help='output directory')
    
    # mode
    parser.add_argument('--batch', action='store_true', help='batch processing mode')
    parser.add_argument('--save-intermediates', '-s', action='store_true', help='save intermediate results')
    parser.add_argument('--max-images', type=int, default=None, help='max images in batch mode')
    
    # rcc parameters
    parser.add_argument('--use-wrcc', action='store_true', help='use wrcc instead of rcc')
    parser.add_argument('--rcc-alpha', type=float, default=0.5, help='rcc alpha (default: 0.5)')
    
    # clahe parameters
    parser.add_argument('--clahe-clip', type=float, default=6.0, help='clahe clip limit (default: 6.0)')
    parser.add_argument('--clahe-tile', type=int, default=32, help='clahe tile size (default: 32)')
    
    # denoise parameters
    parser.add_argument('--denoise-gauss', type=int, default=3, help='gaussian kernel size (default: 3)')
    parser.add_argument('--denoise-median', type=int, default=3, help='median kernel size (default: 3)')
    parser.add_argument('--denoise-bilateral', type=int, default=3, help='bilateral d (default: 3)')
    parser.add_argument('--denoise-sharpen', type=float, default=0.1, help='sharpen strength (default: 0.1)')
    
    # fusion parameters
    parser.add_argument('--fusion-levels', type=int, default=5, help='pyramid levels (default: 5)')
    
    # branch scaling (to adjust contribution of each input)
    parser.add_argument('--scale-rcc', type=float, default=1.0, help='rcc branch scale (default: 1.0)')
    parser.add_argument('--scale-clahe', type=float, default=1.0, help='clahe branch scale (default: 1.0)')
    parser.add_argument('--scale-denoise', type=float, default=0.2, help='denoise branch scale (default: 0.2)')
    
    parser.add_argument('--quiet', '-q', action='store_true', help='minimal output')
    parser.add_argument('--no-eval', action='store_true', help='skip metric evaluation')
    
    args = parser.parse_args()
    
    config = FusionConfig(
        use_wrcc=args.use_wrcc,
        rcc_alpha=args.rcc_alpha,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid_size=(args.clahe_tile, args.clahe_tile),
        denoise_gaussian_ksize=args.denoise_gauss,
        denoise_median_ksize=args.denoise_median,
        denoise_bilateral_d=args.denoise_bilateral,
        denoise_sharpen_strength=args.denoise_sharpen,
        fusion_levels=args.fusion_levels,
        scale_rcc=args.scale_rcc,
        scale_clahe=args.scale_clahe,
        scale_denoise=args.scale_denoise,
    )
    
    verbose = not args.quiet
    
    if args.batch:
        run_batch_processing(
            samples=args.input,
            output_dir=args.output,
            config=config,
            save_intermediates=args.save_intermediates,
            max_images=args.max_images,
            verbose=verbose,
        )
    else:
        paths = collect_image_paths(args.input)
        if not paths:
            print("[error] no valid image files found.")
            return
        
        for path in paths:
            process_single_image(
                input_path=path,
                output_dir=args.output,
                config=config,
                save_intermediates=args.save_intermediates,
                evaluate=not args.no_eval,
                verbose=verbose,
            )


if __name__ == '__main__':
    main()