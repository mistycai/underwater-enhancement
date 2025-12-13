#!/usr/bin/env python3
"""
Batch enhancement of a whole image tree (RUOD-style, no train/val split).

Given an INPUT ROOT like:
    data/RUOD/RUOD_pic/
        train/000001.jpg
        train/000002.jpg
        test/000003.jpg
        ...

This script will enhance ALL images recursively and write them to
an OUTPUT ROOT while preserving the same relative structure:

    input_root  = data/RUOD/RUOD_pic
    output_root = data/RUOD_enhanced_rcc/RUOD_pic   (example)

    data/RUOD_enhanced_rcc/RUOD_pic/
        train/000001.jpg   (enhanced)
        train/000002.jpg
        test/000003.jpg
        ...

It also computes aggregate UIQM / UCIQE / contrast stats over all images.
"""

import os
import sys
import cv2
import json
import argparse
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Import from run_fusion
from run_fusion import (
    FusionConfig,
    run_fusion_pipeline,
    compute_metrics,
)

# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class BatchMetrics:
    """Aggregate metrics across a batch of images."""
    num_images: int = 0
    successful: int = 0
    failed: int = 0

    mean_uiqm_original: float = 0.0
    mean_uiqm_fused: float = 0.0
    mean_uiqm_delta: float = 0.0

    mean_uciqe_original: float = 0.0
    mean_uciqe_fused: float = 0.0
    mean_uciqe_delta: float = 0.0

    mean_contrast_gain: float = 0.0
    mean_entropy_original: float = 0.0
    mean_entropy_fused: float = 0.0

    std_uiqm_delta: float = 0.0
    std_uciqe_delta: float = 0.0
    std_contrast_gain: float = 0.0

    uiqm_improved_count: int = 0
    uciqe_improved_count: int = 0

    min_uiqm_delta: float = 0.0
    max_uiqm_delta: float = 0.0
    min_uciqe_delta: float = 0.0
    max_uciqe_delta: float = 0.0

    per_image_results: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def collect_image_paths(input_root: Path) -> list[Path]:
    """Recursively find all images (case-insensitive extension) under input_root."""
    paths = []
    for p in input_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return sorted(paths)


def save_batch_results_csv(metrics: BatchMetrics, out_dir: Path):
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "batch_results.csv"

    if not metrics.per_image_results:
        return

    with open(csv_path, "w", newline="") as f:
        fieldnames = list(metrics.per_image_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics.per_image_results)

    print(f"Saved per-image results to: {csv_path}")


def save_batch_summary_json(metrics: BatchMetrics, out_dir: Path, config: FusionConfig):
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "num_images": metrics.num_images,
        "successful": metrics.successful,
        "failed": metrics.failed,
        "success_rate": 100.0 * metrics.successful / metrics.num_images
        if metrics.num_images else 0.0,
        "uiqm": {
            "mean_original": metrics.mean_uiqm_original,
            "mean_fused": metrics.mean_uiqm_fused,
            "mean_delta": metrics.mean_uiqm_delta,
            "std_delta": metrics.std_uiqm_delta,
            "min_delta": metrics.min_uiqm_delta,
            "max_delta": metrics.max_uiqm_delta,
            "improved_count": metrics.uiqm_improved_count,
            "improvement_rate": 100.0 * metrics.uiqm_improved_count / metrics.successful
            if metrics.successful else 0.0,
        },
        "uciqe": {
            "mean_original": metrics.mean_uciqe_original,
            "mean_fused": metrics.mean_uciqe_fused,
            "mean_delta": metrics.mean_uciqe_delta,
            "std_delta": metrics.std_uciqe_delta,
            "min_delta": metrics.min_uciqe_delta,
            "max_delta": metrics.max_uciqe_delta,
            "improved_count": metrics.uciqe_improved_count,
            "improvement_rate": 100.0 * metrics.uciqe_improved_count / metrics.successful
            if metrics.successful else 0.0,
        },
        "contrast": {
            "mean_gain": metrics.mean_contrast_gain,
            "std_gain": metrics.std_contrast_gain,
        },
        "entropy": {
            "mean_original": metrics.mean_entropy_original,
            "mean_fused": metrics.mean_entropy_fused,
            "mean_delta": metrics.mean_entropy_fused - metrics.mean_entropy_original,
        },
        "config": {
            "pipeline_mode": config.pipeline_mode,
            "color_correction": config.color_correction,
            "use_wrcc": config.use_wrcc,
            "rcc_alpha": config.rcc_alpha,
            "wb_lambda": config.wb_lambda,
            "clahe_clip_limit": config.clahe_clip_limit,
            "fusion_levels": config.fusion_levels,
        },
    }

    json_path = out_dir / "batch_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved aggregate summary to: {json_path}")


def print_batch_summary(metrics: BatchMetrics, config: FusionConfig):
    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    print(f"Config: mode={config.pipeline_mode}, color={config.color_correction}, "
          f"{'WRCC' if config.use_wrcc else 'RCC'} "
          f"(alpha={config.rcc_alpha}, wb_lambda={config.wb_lambda})")

    print(f"Total images: {metrics.num_images}")
    print(f"Successful:   {metrics.successful}")
    print(f"Failed:       {metrics.failed}")
    if metrics.num_images:
        print(f"Success rate: {100.0 * metrics.successful / metrics.num_images:.1f}%")

    if metrics.successful == 0:
        return

    print(f"\nUIQM:")
    print(f"  Original mean: {metrics.mean_uiqm_original:.4f}")
    print(f"  Fused mean:    {metrics.mean_uiqm_fused:.4f}")
    print(f"  Mean Δ:        {metrics.mean_uiqm_delta:+.4f} ± {metrics.std_uiqm_delta:.4f}")
    print(f"  Improved:      {metrics.uiqm_improved_count}/{metrics.successful}")

    print(f"\nUCIQE:")
    print(f"  Original mean: {metrics.mean_uciqe_original:.2f}")
    print(f"  Fused mean:    {metrics.mean_uciqe_fused:.2f}")
    print(f"  Mean Δ:        {metrics.mean_uciqe_delta:+.2f} ± {metrics.std_uciqe_delta:.2f}")
    print(f"  Improved:      {metrics.uciqe_improved_count}/{metrics.successful}")

    print(f"\nContrast:")
    print(f"  Mean gain:     {metrics.mean_contrast_gain:.4f}x ± {metrics.std_contrast_gain:.4f}")


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def process_folder_with_evaluation(
    input_root: Path,
    output_root: Path,
    config: FusionConfig,
    max_images: Optional[int] = None,
    save_intermediates: bool = False,
    verbose: bool = True,
    resume: bool = False,
) -> BatchMetrics:
    """Enhance all images under input_root, optionally skipping ones already done."""
    all_paths = collect_image_paths(input_root)
    if not all_paths:
        print(f"[ERROR] No images found under {input_root}")
        return BatchMetrics()

    # Build todo list with resume logic
    todo: list[tuple[Path, Path, Path]] = []
    skipped = 0
    for src_path in all_paths:
        rel = src_path.relative_to(input_root)
        out_path = output_root / rel
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        todo.append((src_path, out_path, rel))

    if max_images and max_images < len(todo):
        todo = todo[:max_images]

    if verbose:
        print(f"[Resume] Skipped already enhanced: {skipped}")
        print(f"[Resume] Remaining to process : {len(todo)}")

    if not todo:
        print("[INFO] Nothing left to enhance.")
        return BatchMetrics(num_images=0, successful=0, failed=0)

    output_root.mkdir(parents=True, exist_ok=True)

    # Intermediates
    intermediates_root = output_root.parent / (output_root.name + "_intermediates")
    if save_intermediates:
        for sub in ["color_corrected", "clahe", "denoise", "rcc_only"]:
            (intermediates_root / sub).mkdir(parents=True, exist_ok=True)

    per_image_results = []
    uiqm_originals, uiqm_fused, uiqm_deltas = [], [], []
    uciqe_originals, uciqe_fused, uciqe_deltas = [], [], []
    contrast_gains = []
    entropy_originals, entropy_fused = [], []

    successful = 0
    failed = 0

    iterator = tqdm(todo, desc="Enhancing", unit="img") if verbose else todo

    for src_path, out_path, rel in iterator:
        base = src_path.stem

        try:
            bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if bgr is None:
                failed += 1
                per_image_results.append(
                    {"name": base, "success": False, "error": "Failed to load"}
                )
                continue

            result = run_fusion_pipeline(bgr, config, verbose=False)

            # Save enhanced image with same relative path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), result.fused)

            # Intermediates (same relative structure)
            if save_intermediates:
                rel_jpg = rel.with_suffix(".jpg")
                cv2.imwrite(
                    str(intermediates_root / "color_corrected" / rel_jpg),
                    result.color_corrected,
                )
                cv2.imwrite(
                    str(intermediates_root / "clahe" / rel_jpg),
                    result.clahe,
                )
                cv2.imwrite(
                    str(intermediates_root / "denoise" / rel_jpg),
                    result.denoise,
                )
                if result.rcc_intermediate is not None:
                    cv2.imwrite(
                        str(intermediates_root / "rcc_only" / rel_jpg),
                        result.rcc_intermediate,
                    )

            # Metrics
            orig_m = compute_metrics(result.original, result.original, "Original")
            fused_m = compute_metrics(result.fused, result.original, "Fused")

            uiqm_orig = orig_m.uiqm
            uiqm_fus = fused_m.uiqm
            uiqm_delta = uiqm_fus - uiqm_orig

            uciqe_orig = orig_m.uciqe
            uciqe_fus = fused_m.uciqe
            uciqe_delta = uciqe_fus - uciqe_orig

            contrast_gain = fused_m.contrast

            uiqm_originals.append(uiqm_orig)
            uiqm_fused.append(uiqm_fus)
            uiqm_deltas.append(uiqm_delta)

            uciqe_originals.append(uciqe_orig)
            uciqe_fused.append(uciqe_fus)
            uciqe_deltas.append(uciqe_delta)

            contrast_gains.append(contrast_gain)
            entropy_originals.append(orig_m.entropy_val)
            entropy_fused.append(fused_m.entropy_val)

            per_image_results.append({
                "name": base,
                "success": True,
                "raw_path": str(src_path),
                "enh_path": str(out_path),
                "uiqm_original": uiqm_orig,
                "uiqm_fused": uiqm_fus,
                "uiqm_delta": uiqm_delta,
                "uciqe_original": uciqe_orig,
                "uciqe_fused": uciqe_fus,
                "uciqe_delta": uciqe_delta,
                "contrast_gain": contrast_gain,
                "entropy_original": orig_m.entropy_val,
                "entropy_fused": fused_m.entropy_val,
            })

            successful += 1

        except Exception as e:
            failed += 1
            per_image_results.append(
                {"name": base, "success": False, "error": str(e)}
            )

    metrics = BatchMetrics(
        num_images=len(todo),
        successful=successful,
        failed=failed,
        per_image_results=per_image_results,
    )

    if successful > 0:
        metrics.mean_uiqm_original = float(np.mean(uiqm_originals))
        metrics.mean_uiqm_fused = float(np.mean(uiqm_fused))
        metrics.mean_uiqm_delta = float(np.mean(uiqm_deltas))
        metrics.std_uiqm_delta = float(np.std(uiqm_deltas))
        metrics.min_uiqm_delta = float(np.min(uiqm_deltas))
        metrics.max_uiqm_delta = float(np.max(uiqm_deltas))
        metrics.uiqm_improved_count = sum(1 for d in uiqm_deltas if d > 0)

        metrics.mean_uciqe_original = float(np.mean(uciqe_originals))
        metrics.mean_uciqe_fused = float(np.mean(uciqe_fused))
        metrics.mean_uciqe_delta = float(np.mean(uciqe_deltas))
        metrics.std_uciqe_delta = float(np.std(uciqe_deltas))
        metrics.min_uciqe_delta = float(np.min(uciqe_deltas))
        metrics.max_uciqe_delta = float(np.max(uciqe_deltas))
        metrics.uciqe_improved_count = sum(1 for d in uciqe_deltas if d > 0)

        metrics.mean_contrast_gain = float(np.mean(contrast_gains))
        metrics.std_contrast_gain = float(np.std(contrast_gains))

        metrics.mean_entropy_original = float(np.mean(entropy_originals))
        metrics.mean_entropy_fused = float(np.mean(entropy_fused))

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhance an entire image folder tree (RUOD-style) and keep structure."
    )

    parser.add_argument("--input-root", type=str, required=True,
                        help="Root folder containing raw images (e.g., data/RUOD/RUOD_pic)")
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root folder for enhanced images. "
                             "If omitted, will create sibling with suffix '_enh'.")

    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit total images for quick test.")
    parser.add_argument("--save-intermediates", "-s", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images whose enhanced version already exists and is non-empty.")

    # Pipeline
    parser.add_argument("--mode", choices=["ancuti", "parallel"], default="ancuti")
    parser.add_argument("--color-correction",
                        choices=["gray_world", "rcc", "rcc_wb"],
                        default="rcc_wb")

    # Params (defaults set to your preferred RCC setting)
    parser.add_argument("--rcc-alpha", type=float, default=1.0)
    parser.add_argument("--wb-lambda", type=float, default=0.2)
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--fusion-levels", type=int, default=5)

    parser.add_argument("--use-rcc", action="store_true",
                        help="Use basic RCC instead of WRCC (default: WRCC).")

    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        output_root = input_root.parent / (input_root.name + "_enh")

    verbose = not args.quiet

    if verbose:
        print("=" * 80)
        print("Batch Enhancement (whole folder)")
        print(f"Input root : {input_root}")
        print(f"Output root: {output_root}")
        print("=" * 80)

    # Build config
    config = FusionConfig(
        pipeline_mode=args.mode,
        color_correction=args.color_correction,
        wb_lambda=args.wb_lambda,
        use_wrcc=not args.use_rcc,  # default WRCC unless --use-rcc
        rcc_alpha=args.rcc_alpha,
        clahe_clip_limit=args.clahe_clip,
        fusion_levels=args.fusion_levels,
    )

    metrics = process_folder_with_evaluation(
        input_root=input_root,
        output_root=output_root,
        config=config,
        max_images=args.max_images,
        save_intermediates=args.save_intermediates,
        verbose=verbose,
        resume=args.resume,
    )

    print_batch_summary(metrics, config)

    summaries_dir = output_root / "summaries"
    save_batch_results_csv(metrics, summaries_dir)
    save_batch_summary_json(metrics, summaries_dir, config)

    if verbose:
        print("\nDone.")
        print("Enhanced root:", output_root)


if __name__ == "__main__":
    main()