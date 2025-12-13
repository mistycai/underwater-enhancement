#!/usr/bin/env python3
"""
Component-only ablation (NO fusion, NO "identity" tricks).

Each experiment applies ONLY ONE module:
  - RCC only
  - WRCC only
  - White balance only
  - CLAHE only
  - Denoise only

It evaluates before/after metrics per image and aggregates results.

Usage (from src/):
  python -m ablation_study.ablation_study_components \
      --input-root ../data/RUOD/RUOD_pic/train \
      --experiment wrcc_alpha_only \
      --num-samples 200

  python -m ablation_study.ablation_study_components \
      --input-root ../data/RUOD/RUOD_pic/train \
      --experiment wb_only \
      --num-samples 200
"""

import sys
import cv2
import json
import csv
import argparse
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Callable, Tuple

# -------------------------
# Path setup
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# -------------------------
# Metrics (reuse your project)
# -------------------------
from run_fusion import compute_metrics, METRICS_AVAILABLE

# -------------------------
# TODO: IMPORT YOUR MODULES HERE
# Replace these imports with your real ones.
# -------------------------
# Example expected signatures:
#   apply_rcc_bgr(bgr: np.ndarray, alpha: float) -> np.ndarray
#   apply_wrcc_bgr(bgr: np.ndarray, alpha: float, window_size: int, guided_radius: int) -> np.ndarray
#   gray_world_white_balance(bgr: np.ndarray, lambda_param: float) -> np.ndarray
#   apply_clahe_l_channel(bgr: np.ndarray, clip_limit: float, tile_grid_size: Tuple[int,int],
#                         use_gating: bool, sigma_th: float, ag_th: float) -> np.ndarray
#   apply_denoise(bgr: np.ndarray, bilateral_d: int, sharpen_strength: float, ...) -> np.ndarray

# --- Replace with your real paths ---
from run_fusion import gray_world_white_balance  # if you have it in run_fusion
# from color.rcc import apply_rcc_bgr, apply_wrcc_bgr
# from clahe.clahe import apply_clahe_l_channel
# from denoise.denoise import apply_denoise

# If your functions live inside run_fusion.py, import from there instead:
from run_fusion import apply_rcc_bgr, apply_wrcc_bgr, apply_clahe_l_channel, apply_denoise  # <-- EDIT if needed


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# -------------------------
# Data classes
# -------------------------
@dataclass
class ExperimentResult:
    param_name: str
    param_value: Any
    config_summary: str
    num_images: int = 0
    successful: int = 0
    mean_uiqm_original: float = 0.0
    mean_uciqe_original: float = 0.0
    mean_uiqm_out: float = 0.0
    mean_uciqe_out: float = 0.0
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


# -------------------------
# Sampling
# -------------------------
def collect_and_sample_images(input_root: Path, num_samples: int, seed: int = 42) -> List[Path]:
    all_paths = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    all_paths = sorted(all_paths)
    if not all_paths:
        return []
    if num_samples >= len(all_paths):
        return all_paths
    random.seed(seed)
    return sorted(random.sample(all_paths, num_samples))


# -------------------------
# Component processors (ONLY one applied)
# -------------------------
def proc_rcc_only(bgr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    return apply_rcc_bgr(bgr, alpha=float(p["rcc_alpha"]))

def proc_wrcc_only(bgr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    return apply_wrcc_bgr(
        bgr,
        alpha=float(p["rcc_alpha"]),
        window_size=int(p["rcc_window_size"]),
        guided_radius=int(p["rcc_guided_radius"]),
    )

def proc_wb_only(bgr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    # IMPORTANT: make WB truly no-op at lambda=0.0 inside this function.
    lam = float(p["wb_lambda"])
    if lam <= 0.0:
        return bgr.copy()
    return gray_world_white_balance(bgr, lambda_param=lam)

def proc_clahe_only(bgr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    # CLAHE is applied on luminance only (your detection-aware CLAHE pipeline idea),
    # but since this is "component-only", we apply it directly here.
    return apply_clahe_l_channel(
        bgr,
        clip_limit=float(p["clahe_clip_limit"]),
        tile_grid_size=tuple(p["clahe_tile_grid_size"]),
        use_gating=bool(p["clahe_use_gating"]),
        sigma_th=float(p["clahe_gating_sigma_threshold"]),
        ag_th=float(p["clahe_gating_ag_threshold"]),
    )

def proc_denoise_only(bgr: np.ndarray, p: Dict[str, Any]) -> np.ndarray:
    return apply_denoise(
        bgr,
        bilateral_d=int(p["denoise_bilateral_d"]),
        sharpen_strength=float(p["denoise_sharpen_strength"]),
        gaussian_ksize=int(p.get("denoise_gaussian_ksize", 5)),
        gaussian_sigma=float(p.get("denoise_gaussian_sigma", 1.5)),
        median_ksize=int(p.get("denoise_median_ksize", 3)),
    )


# -------------------------
# Experiment configs (component-only)
# -------------------------
# Each experiment defines:
#   - processor: which single component function to call
#   - param: swept param name
#   - values: swept values list
#   - fixed: other params required by that component
EXPERIMENT_CONFIGS: Dict[str, Dict[str, Any]] = {
    # RCC/WRCC only
    "rcc_alpha_only": {
        "processor": proc_rcc_only,
        "param": "rcc_alpha",
        "values": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "fixed": {},
        "description": "RCC only, sweep alpha",
    },
    "wrcc_alpha_only": {
        "processor": proc_wrcc_only,
        "param": "rcc_alpha",
        "values": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "fixed": {"rcc_window_size": 9, "rcc_guided_radius": 8},
        "description": "WRCC only, sweep alpha",
    },
    "wrcc_window_only": {
        "processor": proc_wrcc_only,
        "param": "rcc_window_size",
        "values": [3, 9, 13, 31],
        "fixed": {"rcc_alpha": 0.5, "rcc_guided_radius": 8},
        "description": "WRCC only, sweep window size",
    },
    "wrcc_radius_only": {
        "processor": proc_wrcc_only,
        "param": "rcc_guided_radius",
        "values": [2, 4, 8, 16],
        "fixed": {"rcc_alpha": 0.5, "rcc_window_size": 9},
        "description": "WRCC only, sweep guided radius",
    },

    # WB only
    "wb_only": {
        "processor": proc_wb_only,
        "param": "wb_lambda",
        "values": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3],
        "fixed": {},
        "description": "White balance only (Gray-world), sweep lambda",
    },

    # CLAHE only
    "clahe_only_clip": {
        "processor": proc_clahe_only,
        "param": "clahe_clip_limit",
        "values": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "fixed": {"clahe_tile_grid_size": (8, 8), "clahe_use_gating": False,
                  "clahe_gating_sigma_threshold": 0.18, "clahe_gating_ag_threshold": 0.06},
        "description": "CLAHE only, sweep clip limit",
    },
    "clahe_only_tile": {
        "processor": proc_clahe_only,
        "param": "clahe_tile_grid_size",
        "values": [(4, 4), (8, 8), (16, 16), (32, 32)],
        "fixed": {"clahe_clip_limit": 2.0, "clahe_use_gating": False,
                  "clahe_gating_sigma_threshold": 0.18, "clahe_gating_ag_threshold": 0.06},
        "description": "CLAHE only, sweep tile grid size",
    },
    "clahe_only_gating": {
        "processor": proc_clahe_only,
        "param": "clahe_use_gating",
        "values": [False, True],
        "fixed": {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": (8, 8),
                  "clahe_gating_sigma_threshold": 0.18, "clahe_gating_ag_threshold": 0.06},
        "description": "CLAHE only, gating off vs on",
    },
    "clahe_only_sigma": {
        "processor": proc_clahe_only,
        "param": "clahe_gating_sigma_threshold",
        "values": [0.10, 0.15, 0.18, 0.20, 0.25],
        "fixed": {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": (8, 8),
                  "clahe_use_gating": True, "clahe_gating_ag_threshold": 0.06},
        "description": "CLAHE only, sweep sigma_L threshold",
    },
    "clahe_only_ag": {
        "processor": proc_clahe_only,
        "param": "clahe_gating_ag_threshold",
        "values": [0.04, 0.06, 0.08, 0.10],
        "fixed": {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": (8, 8),
                  "clahe_use_gating": True, "clahe_gating_sigma_threshold": 0.18},
        "description": "CLAHE only, sweep AG threshold",
    },

    # Denoise only
    "denoise_only_bilateral": {
        "processor": proc_denoise_only,
        "param": "denoise_bilateral_d",
        "values": [0, 3, 5, 7, 9, 11],
        "fixed": {"denoise_sharpen_strength": 0.0, "denoise_gaussian_ksize": 5,
                  "denoise_gaussian_sigma": 1.5, "denoise_median_ksize": 3},
        "description": "Denoise only, sweep bilateral diameter",
    },
    "denoise_only_sharpen": {
        "processor": proc_denoise_only,
        "param": "denoise_sharpen_strength",
        "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "fixed": {"denoise_bilateral_d": 7, "denoise_gaussian_ksize": 5,
                  "denoise_gaussian_sigma": 1.5, "denoise_median_ksize": 3},
        "description": "Denoise only, sweep sharpen strength",
    },
}


# -------------------------
# Core runner
# -------------------------
def run_single_config(
    image_paths: List[Path],
    processor: Callable[[np.ndarray, Dict[str, Any]], np.ndarray],
    params: Dict[str, Any],
    param_name: str,
    param_value: Any,
    verbose: bool = True
) -> ExperimentResult:
    import time

    uiqm_orig, uiqm_out = [], []
    uciqe_orig, uciqe_out = [], []
    contrast_gains = []
    successful = 0
    total_time = 0.0

    desc = f"{param_name}={param_value}"
    iterator = tqdm(image_paths, desc=desc, leave=False) if verbose else image_paths

    for img_path in iterator:
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        try:
            start = time.time()
            out = processor(bgr, params)
            total_time += time.time() - start

            if METRICS_AVAILABLE:
                orig_m = compute_metrics(bgr, bgr, "Original")
                out_m  = compute_metrics(out, bgr, "Out")  # use original as reference if your metric code expects it

                uiqm_orig.append(orig_m.uiqm)
                uiqm_out.append(out_m.uiqm)
                uciqe_orig.append(orig_m.uciqe)
                uciqe_out.append(out_m.uciqe)
                contrast_gains.append(out_m.contrast)

            successful += 1
        except Exception as e:
            if verbose:
                print(f"  Error: {img_path.name}: {e}")

    exp = ExperimentResult(
        param_name=param_name,
        param_value=param_value,
        config_summary=str(params),
        num_images=len(image_paths),
        successful=successful,
    )

    if successful > 0 and METRICS_AVAILABLE:
        uiqm_d = [o - i for o, i in zip(uiqm_out, uiqm_orig)]
        uciqe_d = [o - i for o, i in zip(uciqe_out, uciqe_orig)]

        exp.mean_uiqm_original = float(np.mean(uiqm_orig))
        exp.mean_uiqm_out = float(np.mean(uiqm_out))
        exp.mean_uiqm_delta = float(np.mean(uiqm_d))
        exp.std_uiqm_delta = float(np.std(uiqm_d))

        exp.mean_uciqe_original = float(np.mean(uciqe_orig))
        exp.mean_uciqe_out = float(np.mean(uciqe_out))
        exp.mean_uciqe_delta = float(np.mean(uciqe_d))
        exp.std_uciqe_delta = float(np.std(uciqe_d))

        exp.mean_contrast_gain = float(np.mean(contrast_gains)) if contrast_gains else 0.0
        exp.std_contrast_gain = float(np.std(contrast_gains)) if contrast_gains else 0.0

        exp.uiqm_improved_rate = 100.0 * sum(1 for d in uiqm_d if d > 0) / len(uiqm_d)
        exp.uciqe_improved_rate = 100.0 * sum(1 for d in uciqe_d if d > 0) / len(uciqe_d)

        exp.processing_time_per_image = total_time / successful

    return exp


def run_ablation_study(experiment_name: str, image_paths: List[Path], output_dir: Path, verbose: bool = True) -> AblationStudy:
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    cfg = EXPERIMENT_CONFIGS[experiment_name]
    study = AblationStudy(
        experiment_name=experiment_name,
        description=cfg["description"],
        param_name=cfg["param"],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        num_samples=len(image_paths),
    )

    if verbose:
        print(f"\n{'='*80}")
        print(f"Ablation (component-only): {experiment_name}")
        print(f"Param: {cfg['param']}, Values: {cfg['values']}")
        print(f"Samples: {len(image_paths)}")
        print(f"{'='*80}")

    for val in cfg["values"]:
        params = dict(cfg.get("fixed", {}))
        params[cfg["param"]] = val
        result = run_single_config(
            image_paths=image_paths,
            processor=cfg["processor"],
            params=params,
            param_name=cfg["param"],
            param_value=val,
            verbose=verbose,
        )
        study.results.append(result)

        if verbose:
            print(
                f"  {cfg['param']}={val}: "
                f"UIQM={result.mean_uiqm_out:.4f} (Δ={result.mean_uiqm_delta:+.4f}), "
                f"UCIQE={result.mean_uciqe_out:.2f} (Δ={result.mean_uciqe_delta:+.2f})"
            )

    return study


# -------------------------
# Save outputs (JSON/CSV/LaTeX)
# -------------------------
def save_study_json(study: AblationStudy, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "experiment": study.experiment_name,
        "description": study.description,
        "param": study.param_name,
        "timestamp": study.timestamp,
        "num_samples": study.num_samples,
        "results": [
            {
                "value": str(r.param_value),
                "uiqm": r.mean_uiqm_out,
                "uciqe": r.mean_uciqe_out,
                "contrast": r.mean_contrast_gain,
                "uiqm_delta": r.mean_uiqm_delta,
                "uciqe_delta": r.mean_uciqe_delta,
            }
            for r in study.results
        ],
    }
    out = output_dir / f"{study.experiment_name}_results.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out}")


def save_study_csv(study: AblationStudy, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{study.experiment_name}_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([study.param_name, "UIQM", "UCIQE", "Contrast", "UIQM_delta", "UCIQE_delta"])
        for r in study.results:
            w.writerow([
                str(r.param_value),
                f"{r.mean_uiqm_out:.4f}",
                f"{r.mean_uciqe_out:.4f}",
                f"{r.mean_contrast_gain:.4f}",
                f"{r.mean_uiqm_delta:+.4f}",
                f"{r.mean_uciqe_delta:+.2f}",
            ])
    print(f"Saved: {out}")


def generate_latex_table(study: AblationStudy, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    param_disp = {
        "rcc_alpha": r"$\alpha$",
        "wb_lambda": r"$\lambda$",
        "clahe_clip_limit": "clip\\_limit",
        "clahe_tile_grid_size": "tile\\_size",
        "clahe_use_gating": "gating",
        "clahe_gating_sigma_threshold": r"$\sigma_L$ th",
        "clahe_gating_ag_threshold": "AG th",
        "rcc_window_size": "window\\_size",
        "rcc_guided_radius": "guided\\_radius",
        "denoise_bilateral_d": "bilateral\\_d",
        "denoise_sharpen_strength": "sharpen",
    }.get(study.param_name, study.param_name)

    best_uiqm = max(range(len(study.results)), key=lambda i: study.results[i].mean_uiqm_out)
    best_uciqe = max(range(len(study.results)), key=lambda i: study.results[i].mean_uciqe_out)
    best_contrast = max(range(len(study.results)), key=lambda i: study.results[i].mean_contrast_gain)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{study.description} on {study.num_samples} images.}}",
        f"\\label{{tab:{study.experiment_name}}}",
        r"\begin{tabular}{cccc}",
        r"\hline",
        f"{param_disp} & UIQM & UCIQE & Contrast gain \\\\",
        r"\hline",
    ]

    for i, r in enumerate(study.results):
        v = str(r.param_value)
        u = f"{r.mean_uiqm_out:.4f}"
        c = f"{r.mean_uciqe_out:.4f}"
        g = f"{r.mean_contrast_gain:.4f}"

        if i == best_uiqm: u = r"\textbf{" + u + "}"
        if i == best_uciqe: c = r"\textbf{" + c + "}"
        if i == best_contrast: g = r"\textbf{" + g + "}"

        lines.append(f"{v} & {u} & {c} & {g} \\\\")

    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])

    out = output_dir / f"{study.experiment_name}_table.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Component-only ablations (no fusion)")
    parser.add_argument("--input-root", type=str, required=True, help="Image folder")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--experiment", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    args = parser.parse_args()

    if args.list_experiments:
        print("\nAvailable component-only experiments:")
        for n, c in EXPERIMENT_CONFIGS.items():
            print(f"  {n:25} - {c['param']}: {c['values']}")
        return

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else SCRIPT_DIR / "ablation_results_components"
    verbose = not args.quiet

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

        study = run_ablation_study(exp, image_paths, output_dir, verbose)
        save_study_json(study, output_dir)
        save_study_csv(study, output_dir)
        generate_latex_table(study, output_dir)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()