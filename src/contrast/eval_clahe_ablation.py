#!/usr/bin/env python3
"""
CLAHE Ablation Study Script (high-standard, reproducible sampling + compute_metrics)

Location (suggested):
  src/contrast/eval_clahe_ablation.py

What it does:
  - Samples a fixed, reproducible subset of images (or loads a saved sample list)
  - Runs CLAHE-only enhancement via enhance_wrappers.apply_clahe_bgr (NO other modules)
  - Computes metrics using metrics.compute_metrics.compute_metrics (UIQM/UCIQE/contrast/entropy/colorfulness/avg_gradient)
  - Reports mean metrics + deltas vs raw
  - Optionally saves enhanced images in organized folders
  - Optionally exports CSV

Run:
  python3 -m src.contrast.eval_clahe_ablation --val_dir ./data/RUOD/RUOD_pic/train --experiment clip --num-samples 200 --seed 42
  python3 -m src.contrast.eval_clahe_ablation --val_dir ./data/RUOD/RUOD_pic/train --experiment gating_sigma --sample-list ./ablation_samples_200_seed42.txt
"""

import argparse
import csv
import glob
import os
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..enhance_wrappers import apply_clahe_bgr
from ..metrics.compute_metrics import compute_metrics


# ----------------------------
# IO utilities
# ----------------------------
def list_images(root_dir: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext.upper()}"), recursive=True))
    return sorted(set(files))


def sample_images(
    image_paths: List[str],
    num_samples: Optional[int],
    seed: int = 42,
    sample_list: Optional[str] = None,
    save_sample_list: Optional[str] = None,
) -> List[str]:
    if sample_list:
        if not os.path.isfile(sample_list):
            raise FileNotFoundError(f"--sample-list not found: {sample_list}")
        with open(sample_list, "r") as f:
            chosen = [line.strip() for line in f if line.strip()]
        chosen = [p for p in chosen if os.path.isfile(p)]
        if not chosen:
            raise RuntimeError(f"--sample-list is empty or all files missing: {sample_list}")
        return chosen

    if num_samples is None or num_samples >= len(image_paths):
        chosen = image_paths
    else:
        rng = random.Random(seed)
        chosen = sorted(rng.sample(image_paths, num_samples))

    if save_sample_list:
        os.makedirs(os.path.dirname(save_sample_list) or ".", exist_ok=True)
        with open(save_sample_list, "w") as f:
            for p in chosen:
                f.write(p + "\n")

    return chosen


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def cfg_name_from_params(
    clip: float,
    tile_grid: Tuple[int, int],
    gating: bool,
    sigma: float,
    ag: float,
) -> str:
    # stable short naming for folders
    return f"clip{clip:.2f}_tile{tile_grid[0]}x{tile_grid[1]}_g{int(gating)}_s{sigma:.3f}_ag{ag:.3f}"


def save_enhanced_image(
    enhanced_bgr: np.ndarray,
    orig_path: str,
    val_root: str,
    out_root: str,
    cfg_name: str,
) -> None:
    rel_path = os.path.relpath(orig_path, val_root)
    rel_dir = os.path.dirname(rel_path)
    base = os.path.splitext(os.path.basename(rel_path))[0]
    out_dir = os.path.join(out_root, cfg_name, rel_dir)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, base + ".jpg")
    cv2.imwrite(out_path, enhanced_bgr)


# ----------------------------
# Metrics aggregation
# ----------------------------
def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def evaluate_clahe_setting(
    image_paths: List[str],
    val_root: str,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
    use_gating: bool,
    sigma_threshold: float,
    ag_threshold: float,
    save_imgs: bool = False,
    out_root: Optional[str] = None,
) -> Dict[str, float]:
    """
    CLAHE-only evaluation:
      raw_m = compute_metrics(raw, raw)
      clahe = apply_clahe_bgr(raw, ...)
      enh_m = compute_metrics(clahe, raw)
    """
    if save_imgs and out_root is None:
        raise ValueError("out_root must be provided when save_imgs=True")

    uiqm_vals, uciqe_vals, contrast_vals = [], [], []
    entropy_vals, colorful_vals, avggrad_vals = [], [], []

    uiqm_deltas, uciqe_deltas, contrast_deltas = [], [], []

    # Optional: also record gating/clahe internal info rates
    applied_flags: List[int] = []
    gains: List[float] = []

    cfg_name = cfg_name_from_params(clip_limit, tile_grid_size, use_gating, sigma_threshold, ag_threshold)

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue

        raw_m = compute_metrics(bgr, bgr, name="raw")

        enh_bgr, info = apply_clahe_bgr(
            bgr,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            use_gating=use_gating,
            sigma_threshold=sigma_threshold,
            ag_threshold=ag_threshold,
        )

        enh_m = compute_metrics(enh_bgr, bgr, name="clahe")

        uiqm_vals.append(enh_m.uiqm)
        uciqe_vals.append(enh_m.uciqe)
        contrast_vals.append(enh_m.contrast)
        entropy_vals.append(enh_m.entropy_val)
        colorful_vals.append(enh_m.colorfulness_val)
        avggrad_vals.append(enh_m.avg_gradient_val)

        uiqm_deltas.append(enh_m.uiqm - raw_m.uiqm)
        uciqe_deltas.append(enh_m.uciqe - raw_m.uciqe)
        contrast_deltas.append(enh_m.contrast - 1.0)

        applied_flags.append(int(bool(info.get("applied", True))))
        gains.append(float(info.get("contrast_gain", 1.0)))

        if save_imgs:
            save_enhanced_image(
                enhanced_bgr=enh_bgr,
                orig_path=path,
                val_root=val_root,
                out_root=out_root,
                cfg_name=cfg_name,
            )

    n = len(uiqm_vals)
    if n == 0:
        return {
            "num_images_used": 0,
            "uiqm": 0.0, "uciqe": 0.0, "contrast_gain": 0.0,
            "entropy": 0.0, "colorfulness": 0.0, "avg_gradient": 0.0,
            "uiqm_delta": 0.0, "uciqe_delta": 0.0, "contrast_delta": 0.0,
            "clahe_applied_rate": 0.0, "clahe_contrast_gain_mean": 0.0,
        }

    return {
        "num_images_used": int(n),
        "uiqm": _mean(uiqm_vals),
        "uciqe": _mean(uciqe_vals),
        "contrast_gain": _mean(contrast_vals),
        "entropy": _mean(entropy_vals),
        "colorfulness": _mean(colorful_vals),
        "avg_gradient": _mean(avggrad_vals),
        "uiqm_delta": _mean(uiqm_deltas),
        "uciqe_delta": _mean(uciqe_deltas),
        "contrast_delta": _mean(contrast_deltas),
        "clahe_applied_rate": 100.0 * float(np.mean(applied_flags)),
        "clahe_contrast_gain_mean": _mean(gains),
    }


# ----------------------------
# Experiments
# ----------------------------
def get_experiment_configs(experiment: str) -> List[Dict[str, Any]]:
    """
    Returns a list of configs to sweep.
    Each config is a dict with keys:
      clip_limit, tile_grid_size, use_gating, sigma_threshold, ag_threshold
    """
    # Base defaults (match your wrapper defaults)
    base = dict(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        use_gating=False,
        sigma_threshold=0.18,
        ag_threshold=0.06,
    )

    cfgs: List[Dict[str, Any]] = []

    if experiment == "clip":
        for clip in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
            c = base.copy()
            c["clip_limit"] = clip
            cfgs.append(c)

    elif experiment == "tile":
        for tile in [(4, 4), (8, 8), (16, 16), (32, 32)]:
            c = base.copy()
            c["tile_grid_size"] = tile
            cfgs.append(c)

    elif experiment == "gating":
        for g in [False, True]:
            c = base.copy()
            c["use_gating"] = g
            cfgs.append(c)

    elif experiment == "gating_sigma":
        # gating must be enabled, sweep sigma
        for s in [0.10, 0.15, 0.18, 0.20, 0.25]:
            c = base.copy()
            c["use_gating"] = True
            c["sigma_threshold"] = s
            cfgs.append(c)

    elif experiment == "gating_ag":
        # gating must be enabled, sweep AG threshold
        for ag in [0.04, 0.06, 0.08, 0.10]:
            c = base.copy()
            c["use_gating"] = True
            c["ag_threshold"] = ag
            cfgs.append(c)

    elif experiment == "full_gating_grid":
        # higher-standard: small grid for (sigma, ag) jointly with gating on
        sigmas = [0.15, 0.18, 0.20]
        ags = [0.04, 0.06, 0.08]
        for s in sigmas:
            for ag in ags:
                c = base.copy()
                c["use_gating"] = True
                c["sigma_threshold"] = s
                c["ag_threshold"] = ag
                cfgs.append(c)

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    return cfgs


def cfg_to_row(cfg: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    row = {}
    row.update(cfg)
    row.update(metrics)
    # normalize tuple for csv readability
    tile = row.get("tile_grid_size", (8, 8))
    if isinstance(tile, tuple):
        row["tile_grid_size"] = f"{tile[0]}x{tile[1]}"
    return row


def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLAHE-only ablation study (compute_metrics)")
    p.add_argument("--val_dir", type=str, required=True)

    p.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["clip", "tile", "gating", "gating_sigma", "gating_ag", "full_gating_grid"],
        help="Which CLAHE parameter sweep to run.",
    )

    p.add_argument("--save_imgs", action="store_true")
    p.add_argument("--out_dir", type=str, default="results/clahe_ablation")

    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-list", type=str, default=None)
    p.add_argument("--save-sample-list", type=str, default=None)

    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    all_paths = list_images(args.val_dir)
    print(f"[INFO] Found {len(all_paths)} images in {args.val_dir}")
    if not all_paths:
        print("[ERROR] No images found.")
        return

    image_paths = sample_images(
        image_paths=all_paths,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_list=args.sample_list,
        save_sample_list=args.save_sample_list,
    )
    print(f"[INFO] Using {len(image_paths)} images for evaluation.")

    cfgs = get_experiment_configs(args.experiment)
    print(f"[INFO] Running experiment '{args.experiment}' with {len(cfgs)} settings...")

    rows: List[Dict[str, Any]] = []

    # Baseline (raw) once, for sanity prints (optional)
    # You still compute per-image raw metrics inside evaluate_clahe_setting for deltas,
    # but this baseline helps you quickly see absolute values.
    raw_uiqm, raw_uciqe = [], []
    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        m = compute_metrics(bgr, bgr, name="raw")
        raw_uiqm.append(m.uiqm)
        raw_uciqe.append(m.uciqe)
    if raw_uiqm:
        print(f"[BASELINE] raw mean UIQM={float(np.mean(raw_uiqm)):.4f}, UCIQE={float(np.mean(raw_uciqe)):.2f}")

    # Sweep
    for i, cfg in enumerate(cfgs):
        metrics = evaluate_clahe_setting(
            image_paths=image_paths,
            val_root=args.val_dir,
            clip_limit=cfg["clip_limit"],
            tile_grid_size=cfg["tile_grid_size"],
            use_gating=cfg["use_gating"],
            sigma_threshold=cfg["sigma_threshold"],
            ag_threshold=cfg["ag_threshold"],
            save_imgs=args.save_imgs,
            out_root=args.out_dir if args.save_imgs else None,
        )
        row = cfg_to_row(cfg, metrics)
        rows.append(row)

        print(
            f"[{i+1:02d}/{len(cfgs):02d}] "
            f"clip={cfg['clip_limit']:.2f}, tile={cfg['tile_grid_size'][0]}x{cfg['tile_grid_size'][1]}, "
            f"g={int(cfg['use_gating'])}, s={cfg['sigma_threshold']:.3f}, ag={cfg['ag_threshold']:.3f} | "
            f"UIQM={metrics['uiqm']:.4f} (Δ={metrics['uiqm_delta']:+.4f}), "
            f"UCIQE={metrics['uciqe']:.2f} (Δ={metrics['uciqe_delta']:+.2f}), "
            f"CG={metrics['contrast_gain']:.4f} (Δ={metrics['contrast_delta']:+.4f}), "
            f"applied={metrics['clahe_applied_rate']:.1f}%"
        )

    # Rank + print top by UIQM
    rows_sorted = sorted(rows, key=lambda r: r.get("uiqm", 0.0), reverse=True)
    print("\n======================")
    print("Top configs by UIQM")
    print("======================")
    for r in rows_sorted[:10]:
        print(r)

    # Optional CSV
    if args.csv:
        write_csv(rows_sorted, args.csv)
        print(f"[INFO] Saved CSV to: {args.csv}")


if __name__ == "__main__":
    main()