#!/usr/bin/env python3
"""
Ablation Study Script for Underwater Image Enhancement

Location: src/ablation_study/ablation_study.py

This script runs systematic experiments to find optimal parameters for each
component of the enhancement pipeline. Results are saved in LaTeX-ready format.

Usage:
    # From src/ directory:
    python -m ablation_study.ablation_study --input-root ../data/RUOD/RUOD_pic/train --experiment wrcc_alpha --num-samples 200

    # Or run directly from ablation_study folder:
    cd ablation_study
    python ablation_study.py --input-root ../../data/RUOD/RUOD_pic/train --num-samples 200 --experiment wrcc_alpha
"""

import os
import sys
import cv2
import json
import argparse
import random
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product

# Path Setup
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Import from your project
from run_fusion import FusionConfig, run_fusion_pipeline, compute_metrics, METRICS_AVAILABLE

# Sample size recommendations
RECOMMENDED_SAMPLES = {'quick': 50, 'ablation': 200, 'thorough': 500, 'full': 1000}

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'rcc_alpha': {
        'param': 'rcc_alpha', 'values': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        'fixed': {'color_correction': 'rcc', 'use_wrcc': False, 'wb_lambda': 0.0},
        'description': 'RCC with different alpha values',
    },
    'wrcc_alpha': {
        'param': 'rcc_alpha', 'values': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        'fixed': {'color_correction': 'rcc', 'use_wrcc': True, 'wb_lambda': 0.0, 'rcc_window_size': 9, 'rcc_guided_radius': 8},
        'description': 'WRCC with different alpha values',
    },
    'wrcc_window': {
        'param': 'rcc_window_size', 'values': [3, 9, 13, 31],
        'fixed': {'color_correction': 'rcc', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.0, 'rcc_guided_radius': 8},
        'description': 'WRCC window_size ablation',
    },
    'wrcc_radius': {
        'param': 'rcc_guided_radius', 'values': [2, 4, 8, 16],
        'fixed': {'color_correction': 'rcc', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.0, 'rcc_window_size': 9},
        'description': 'WRCC guided_radius ablation',
    },
    'wb_lambda': {
        'param': 'wb_lambda', 'values': [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 
                  'rcc_window_size': 9, 'rcc_guided_radius': 2},
        'description': 'White balance lambda ablation (with WRCC alpha=0.5)',
    },
    'wb_lambda_only': {
        'param': 'wb_lambda', 'values': [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        'fixed': {'color_correction': 'gray_world'},
        'description': 'Gray-World only (no RCC) lambda ablation',
    },
    'wrcc_wb_combined': {
        'param': 'wb_lambda', 'values': [0.0, 0.1, 0.15, 0.2, 0.25],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'rcc_window_size': 9, 'rcc_guided_radius': 2},
        'description': 'WRCC + WB combined',
    },
    'color_method': {
        'param': 'color_correction', 'values': ['gray_world', 'rcc', 'rcc_wb'],
        'fixed': {'rcc_alpha': 0.5, 'wb_lambda': 0.2, 'use_wrcc': True},
        'description': 'Color correction method comparison',
    },
    'clahe_clip': {
        'param': 'clahe_clip_limit', 'values': [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2},
        'description': 'CLAHE clip_limit ablation',
    },
    'clahe_tile': {
        'param': 'clahe_tile_grid_size', 'values': [(4,4), (8,8), (16,16), (32,32)],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2, 'clahe_clip_limit': 2.0},
        'description': 'CLAHE tile_size ablation',
    },
    'clahe_gating': {
        'param': 'clahe_use_gating', 'values': [False, True],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2, 
                  'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8,8)},
        'description': 'CLAHE gating ON vs OFF',
    },
    'clahe_gating_sigma': {
        'param': 'clahe_gating_sigma_threshold', 'values': [0.10, 0.15, 0.18, 0.20, 0.25],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2,
                  'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8,8), 'clahe_use_gating': True},
        'description': 'CLAHE gating sigma_L threshold',
    },
    'clahe_gating_ag': {
        'param': 'clahe_gating_ag_threshold', 'values': [0.04, 0.06, 0.08, 0.10],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2,
                  'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8,8), 'clahe_use_gating': True,
                  'clahe_gating_sigma_threshold': 0.18},
        'description': 'CLAHE gating AG threshold (sigma=0.18)',
    },
    'denoise_bilateral': {
        'param': 'denoise_bilateral_d', 'values': [0, 3, 5, 7, 9, 11],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2},
        'description': 'Bilateral filter diameter',
    },
    'denoise_sharpen': {
        'param': 'denoise_sharpen_strength', 'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2, 'denoise_bilateral_d': 7},
        'description': 'Sharpening strength',
    },
    'fusion_levels': {
        'param': 'fusion_levels', 'values': [3, 4, 5, 6, 7],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2},
        'description': 'Fusion pyramid levels',
    },
    'fusion_mode': {
        'param': 'pipeline_mode', 'values': ['ancuti', 'parallel'],
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2},
        'description': '2-input vs 3-input fusion',
    },
}

GRID_SEARCH_CONFIGS = {
    'color_grid': {
        'params': ['rcc_alpha', 'wb_lambda'],
        'ranges': {'rcc_alpha': [0.3, 0.5, 0.7, 1.0], 'wb_lambda': [0.1, 0.15, 0.2, 0.25]},
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True},
    },
    'clahe_grid': {
        'params': ['clahe_clip_limit', 'clahe_tile_grid_size'],
        'ranges': {'clahe_clip_limit': [1.5, 2.0, 2.5], 'clahe_tile_grid_size': [(4,4), (8,8), (16,16)]},
        'fixed': {'color_correction': 'rcc_wb', 'use_wrcc': True, 'rcc_alpha': 0.5, 'wb_lambda': 0.2},
    },
}

@dataclass
class ExperimentResult:
    param_name: str
    param_value: Any
    config_summary: str
    num_images: int = 0
    successful: int = 0
    mean_uiqm_original: float = 0.0
    mean_uciqe_original: float = 0.0
    mean_uiqm_fused: float = 0.0
    mean_uciqe_fused: float = 0.0
    mean_contrast_gain: float = 0.0
    mean_uiqm_delta: float = 0.0
    std_uiqm_delta: float = 0.0
    mean_uciqe_delta: float = 0.0
    std_uciqe_delta: float = 0.0
    std_contrast_gain: float = 0.0
    uiqm_improved_rate: float = 0.0
    uciqe_improved_rate: float = 0.0
    processing_time_per_image: float = 0.0

@dataclass
class AblationStudy:
    experiment_name: str
    description: str
    param_name: str
    timestamp: str
    num_samples: int
    results: List[ExperimentResult] = field(default_factory=list)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collect_and_sample_images(input_root: Path, num_samples: int, seed: int = 42) -> List[Path]:
    all_paths = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    all_paths = sorted(all_paths)
    if not all_paths:
        return []
    if num_samples >= len(all_paths):
        return all_paths
    random.seed(seed)
    return sorted(random.sample(all_paths, num_samples))

def build_config(base_config: Dict[str, Any], param_name: str, param_value: Any) -> FusionConfig:
    config_dict = {
        'pipeline_mode': 'ancuti', 'color_correction': 'rcc_wb', 'use_wrcc': True,
        'rcc_alpha': 0.5, 'wb_lambda': 0.2, 'rcc_window_size': 9, 'rcc_guided_radius': 8,
        'clahe_clip_limit': 2.0, 'clahe_tile_grid_size': (8, 8), 'clahe_use_gating': False,
        'clahe_gating_sigma_threshold': 0.18, 'clahe_gating_ag_threshold': 0.06,
        'denoise_gaussian_ksize': 5, 'denoise_gaussian_sigma': 1.5, 'denoise_median_ksize': 3,
        'denoise_bilateral_d': 7, 'denoise_sharpen_strength': 0.3, 'fusion_levels': 5,
        'lam_laplacian': 1.0, 'lam_local_contrast': 1.0, 'lam_saliency': 1.0,
        'lam_exposedness': 1.0, 'exposedness_sigma': 0.25,
    }
    config_dict.update(base_config)
    config_dict[param_name] = param_value
    if config_dict.get('color_correction') == 'gray_world':
        config_dict['rcc_alpha'] = 0.0
    # Filter to only FusionConfig fields
    valid_fields = {'pipeline_mode', 'color_correction', 'use_wrcc', 'rcc_alpha', 'wb_lambda',
                   'rcc_window_size', 'rcc_guided_radius', 'clahe_clip_limit', 'clahe_tile_grid_size',
                   'clahe_use_gating', 'denoise_gaussian_ksize', 'denoise_gaussian_sigma',
                   'denoise_median_ksize', 'denoise_bilateral_d', 'denoise_sharpen_strength',
                   'fusion_levels', 'lam_laplacian', 'lam_local_contrast', 'lam_saliency',
                   'lam_exposedness', 'exposedness_sigma'}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    return FusionConfig(**filtered_config)

def run_single_config(image_paths, config, param_name, param_value, verbose=True):
    import time
    uiqm_orig, uiqm_fused, uciqe_orig, uciqe_fused, contrast_gains = [], [], [], [], []
    successful, total_time = 0, 0.0
    
    desc = f"{param_name}={param_value}" if not isinstance(param_value, tuple) else f"{param_name}={param_value[0]}x{param_value[1]}"
    iterator = tqdm(image_paths, desc=desc, leave=False) if verbose else image_paths
    
    for img_path in iterator:
        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None: continue
            start = time.time()
            result = run_fusion_pipeline(bgr, config, verbose=False)
            total_time += time.time() - start
            if METRICS_AVAILABLE:
                orig_m = compute_metrics(result.original, result.original, "Original")
                fused_m = compute_metrics(result.fused, result.original, "Fused")
                uiqm_orig.append(orig_m.uiqm); uiqm_fused.append(fused_m.uiqm)
                uciqe_orig.append(orig_m.uciqe); uciqe_fused.append(fused_m.uciqe)
                contrast_gains.append(fused_m.contrast)
            successful += 1
        except Exception as e:
            if verbose: print(f"  Error: {img_path.name}: {e}")
    
    exp = ExperimentResult(param_name=param_name, param_value=param_value,
                          config_summary=f"{config.color_correction}, α={config.rcc_alpha}, λ={config.wb_lambda}",
                          num_images=len(image_paths), successful=successful)
    
    if successful > 0 and METRICS_AVAILABLE:
        uiqm_d = [f-o for f,o in zip(uiqm_fused, uiqm_orig)]
        uciqe_d = [f-o for f,o in zip(uciqe_fused, uciqe_orig)]
        exp.mean_uiqm_original = float(np.mean(uiqm_orig))
        exp.mean_uiqm_fused = float(np.mean(uiqm_fused))
        exp.mean_uiqm_delta = float(np.mean(uiqm_d))
        exp.std_uiqm_delta = float(np.std(uiqm_d))
        exp.mean_uciqe_original = float(np.mean(uciqe_orig))
        exp.mean_uciqe_fused = float(np.mean(uciqe_fused))
        exp.mean_uciqe_delta = float(np.mean(uciqe_d))
        exp.std_uciqe_delta = float(np.std(uciqe_d))
        exp.mean_contrast_gain = float(np.mean(contrast_gains))
        exp.std_contrast_gain = float(np.std(contrast_gains))
        exp.uiqm_improved_rate = 100.0 * sum(1 for d in uiqm_d if d > 0) / len(uiqm_d)
        exp.uciqe_improved_rate = 100.0 * sum(1 for d in uciqe_d if d > 0) / len(uciqe_d)
        exp.processing_time_per_image = total_time / successful
    return exp

def run_ablation_study(experiment_name, image_paths, output_dir, verbose=True):
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}")
    cfg = EXPERIMENT_CONFIGS[experiment_name]
    study = AblationStudy(experiment_name=experiment_name, description=cfg['description'],
                         param_name=cfg['param'], timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         num_samples=len(image_paths))
    if verbose:
        print(f"\n{'='*80}\nAblation: {experiment_name}\nParam: {cfg['param']}, Values: {cfg['values']}\nSamples: {len(image_paths)}\n{'='*80}")
    for val in cfg['values']:
        config = build_config(cfg.get('fixed', {}), cfg['param'], val)
        result = run_single_config(image_paths, config, cfg['param'], val, verbose)
        study.results.append(result)
        if verbose:
            v = f"{val[0]}x{val[1]}" if isinstance(val, tuple) else str(val)
            print(f"  {cfg['param']}={v}: UIQM={result.mean_uiqm_fused:.4f} (Δ={result.mean_uiqm_delta:+.4f}), UCIQE={result.mean_uciqe_fused:.2f} (Δ={result.mean_uciqe_delta:+.2f})")
    return study

def save_study_json(study, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {'experiment': study.experiment_name, 'description': study.description, 'param': study.param_name,
            'timestamp': study.timestamp, 'num_samples': study.num_samples,
            'results': [{'value': str(r.param_value), 'uiqm': r.mean_uiqm_fused, 'uciqe': r.mean_uciqe_fused,
                        'contrast': r.mean_contrast_gain, 'uiqm_delta': r.mean_uiqm_delta, 'uciqe_delta': r.mean_uciqe_delta}
                       for r in study.results]}
    with open(output_dir / f"{study.experiment_name}_results.json", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_dir}/{study.experiment_name}_results.json")

def save_study_csv(study, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{study.experiment_name}_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([study.param_name, 'UIQM', 'UCIQE', 'Contrast', 'UIQM_delta', 'UCIQE_delta'])
        for r in study.results:
            v = f"{r.param_value[0]}x{r.param_value[1]}" if isinstance(r.param_value, tuple) else str(r.param_value)
            w.writerow([v, f"{r.mean_uiqm_fused:.4f}", f"{r.mean_uciqe_fused:.4f}", f"{r.mean_contrast_gain:.4f}",
                       f"{r.mean_uiqm_delta:+.4f}", f"{r.mean_uciqe_delta:+.2f}"])
    print(f"Saved: {output_dir}/{study.experiment_name}_results.csv")

def generate_latex_table(study, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    param_disp = {'rcc_alpha': r'$\alpha$', 'wb_lambda': r'$\lambda$', 'clahe_clip_limit': 'clip\\_limit',
                  'clahe_tile_grid_size': 'tile\\_size', 'fusion_levels': 'levels', 'pipeline_mode': 'Mode',
                  'color_correction': 'Method', 'rcc_window_size': 'window\\_size', 'rcc_guided_radius': 'guided\\_radius',
                  'denoise_bilateral_d': 'bilateral\\_d', 'denoise_sharpen_strength': 'sharpen'}.get(study.param_name, study.param_name)
    best_uiqm = max(range(len(study.results)), key=lambda i: study.results[i].mean_uiqm_fused)
    best_uciqe = max(range(len(study.results)), key=lambda i: study.results[i].mean_uciqe_fused)
    best_contrast = max(range(len(study.results)), key=lambda i: study.results[i].mean_contrast_gain)
    
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{{study.description} on {study.num_samples} images.}}",
             f"\\label{{tab:{study.experiment_name}}}", r"\begin{tabular}{cccc}", r"\hline",
             f"{param_disp} & UIQM & UCIQE & Contrast gain \\\\", r"\hline"]
    for i, r in enumerate(study.results):
        v = f"{r.param_value[0]}$\\times${r.param_value[1]}" if isinstance(r.param_value, tuple) else str(r.param_value)
        u = f"{r.mean_uiqm_fused:.4f}"; c = f"{r.mean_uciqe_fused:.4f}"; g = f"{r.mean_contrast_gain:.4f}"
        if i == best_uiqm: u = r"\textbf{" + u + "}"
        if i == best_uciqe: c = r"\textbf{" + c + "}"
        if i == best_contrast: g = r"\textbf{" + g + "}"
        lines.append(f"{v} & {u} & {c} & {g} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    
    with open(output_dir / f"{study.experiment_name}_table.tex", 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {output_dir}/{study.experiment_name}_table.tex")
    print("\nLaTeX Table:\n" + '\n'.join(lines))

def print_study_summary(study):
    print(f"\n{'='*80}\nRESULTS: {study.experiment_name} ({study.num_samples} samples)\n{'='*80}")
    print(f"{study.param_name:>15} | {'UIQM':>10} | {'UCIQE':>10} | {'Contrast':>10} | {'UIQM Δ':>10}")
    print("-"*70)
    for r in study.results:
        v = f"{r.param_value[0]}x{r.param_value[1]}" if isinstance(r.param_value, tuple) else str(r.param_value)
        print(f"{v:>15} | {r.mean_uiqm_fused:>10.4f} | {r.mean_uciqe_fused:>10.4f} | {r.mean_contrast_gain:>10.4f} | {r.mean_uiqm_delta:>+10.4f}")
    best_u = max(study.results, key=lambda r: r.mean_uiqm_fused)
    best_c = max(study.results, key=lambda r: r.mean_uciqe_fused)
    print(f"\nBest UIQM: {study.param_name}={best_u.param_value} → {best_u.mean_uiqm_fused:.4f}")
    print(f"Best UCIQE: {study.param_name}={best_c.param_value} → {best_c.mean_uciqe_fused:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Ablation studies for underwater enhancement")
    parser.add_argument("--input-root", type=str, help="Image folder")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--experiment", nargs='+', default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()
    
    if args.list_experiments:
        print("\nAvailable experiments:")
        for n, c in EXPERIMENT_CONFIGS.items():
            print(f"  {n:20} - {c['param']}: {c['values']}")
        print(f"\nRecommended: --num-samples 200")
        return
    
    if not args.input_root:
        parser.error("--input-root required")
    
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else SCRIPT_DIR / "ablation_results"
    
    print(f"Sampling {args.num_samples} images from {input_root}...")
    image_paths = collect_and_sample_images(input_root, args.num_samples, args.seed)
    if not image_paths:
        print(f"No images found in {input_root}")
        return
    print(f"Sampled {len(image_paths)} images")
    
    if not args.experiment:
        print("No experiment specified. Use --experiment or --list-experiments")
        return
    
    for exp in args.experiment:
        if exp not in EXPERIMENT_CONFIGS:
            print(f"Unknown: {exp}")
            continue
        study = run_ablation_study(exp, image_paths, output_dir, not args.quiet)
        save_study_json(study, output_dir)
        save_study_csv(study, output_dir)
        generate_latex_table(study, output_dir)
        print_study_summary(study)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()