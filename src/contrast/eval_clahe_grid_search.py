import argparse
import csv
import glob
import os
import random
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import cv2
import numpy as np

from ..enhance_wrappers import apply_clahe_bgr
from ..metrics.compute_metrics import compute_metrics

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


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _std(xs: List[float]) -> float:
    return float(np.std(xs)) if xs else 0.0


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else 0.0


def _percentile(xs: List[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else 0.0


def get_grid_configs(
    clips: List[float] = None,
    tiles: List[int] = None,
) -> List[Dict[str, Any]]:
    if clips is None:
        clips = [1.0, 2.0, 3.0, 4.0, 6.0]
    if tiles is None:
        tiles = [8, 16, 32]
    
    cfgs = []
    for clip in clips:
        for tile in tiles:
            cfgs.append({
                'clip_limit': clip,
                'tile_grid_size': (tile, tile),
                'use_gating': False,
                'sigma_threshold': 0.18,
                'ag_threshold': 0.06,
            })
    return cfgs


def cfg_to_name(clip: float, tile: Tuple[int, int]) -> str:
    return f"clip{clip:.1f}_tile{tile[0]}x{tile[1]}"

def evaluate_single_config(
    image_paths: List[str],
    val_root: str,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
    save_imgs: bool = False,
    out_root: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    cfg_name = cfg_to_name(clip_limit, tile_grid_size)
    
    if save_imgs:
        img_dir = os.path.join(out_root, "images", cfg_name)
        ensure_dir(img_dir)

    uiqm_vals, uciqe_vals, contrast_vals = [], [], []
    entropy_vals, colorful_vals, avggrad_vals = [], [], []
    uiqm_deltas, uciqe_deltas = [], []
    gains = []
    
    raw_uiqm_vals, raw_uciqe_vals = [], []
    
    for i, path in enumerate(image_paths):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        
        basename = os.path.splitext(os.path.basename(path))[0]

        raw_m = compute_metrics(bgr, bgr, name="raw")
        raw_uiqm_vals.append(raw_m.uiqm)
        raw_uciqe_vals.append(raw_m.uciqe)

        enh_bgr, info = apply_clahe_bgr(
            bgr,
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
            use_gating=False,
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
        gains.append(float(info.get("contrast_gain", 1.0)))
        
        if save_imgs:
            out_path = os.path.join(img_dir, f"{basename}.jpg")
            cv2.imwrite(out_path, enh_bgr)
    
    n = len(uiqm_vals)
    if n == 0:
        return {"num_images": 0, "config": cfg_name}

    result = {
        "config": cfg_name,
        "clip_limit": clip_limit,
        "tile_size": tile_grid_size[0],
        "num_images": n,
        
        # Raw baseline
        "raw_uiqm_mean": _mean(raw_uiqm_vals),
        "raw_uiqm_std": _std(raw_uiqm_vals),
        "raw_uciqe_mean": _mean(raw_uciqe_vals),
        
        # Enhanced metrics - mean
        "uiqm_mean": _mean(uiqm_vals),
        "uciqe_mean": _mean(uciqe_vals),
        "contrast_mean": _mean(contrast_vals),
        "entropy_mean": _mean(entropy_vals),
        "colorfulness_mean": _mean(colorful_vals),
        "avg_gradient_mean": _mean(avggrad_vals),
        
        # Enhanced metrics - std (for error bars)
        "uiqm_std": _std(uiqm_vals),
        "uciqe_std": _std(uciqe_vals),
        "contrast_std": _std(contrast_vals),
        
        # Enhanced metrics - percentiles (for robustness)
        "uiqm_median": _median(uiqm_vals),
        "uiqm_p25": _percentile(uiqm_vals, 25),
        "uiqm_p75": _percentile(uiqm_vals, 75),
        "uciqe_median": _median(uciqe_vals),
        
        # Deltas from raw
        "uiqm_delta_mean": _mean(uiqm_deltas),
        "uiqm_delta_std": _std(uiqm_deltas),
        "uciqe_delta_mean": _mean(uciqe_deltas),
        "uciqe_delta_std": _std(uciqe_deltas),
        
        # Contrast gain
        "contrast_gain_mean": _mean(gains),
        "contrast_gain_std": _std(gains),
        
        # Win rate (how many images improved vs raw)
        "uiqm_improved_rate": 100.0 * sum(1 for d in uiqm_deltas if d > 0) / n,
        "uciqe_improved_rate": 100.0 * sum(1 for d in uciqe_deltas if d > 0) / n,
    }
    
    return result

def write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    ensure_dir(os.path.dirname(out_path) or ".")
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_summary(rows: List[Dict[str, Any]], out_path: str, args: argparse.Namespace) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    
    rows_by_uiqm = sorted(rows, key=lambda r: r.get("uiqm_mean", 0), reverse=True)

    rows_by_uciqe = sorted(rows, key=lambda r: r.get("uciqe_mean", 0), reverse=True)
    
    with open(out_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("STANDARD CLAHE GRID SEARCH RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {args.val_dir}\n")
        f.write(f"Num samples: {args.num_samples}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Clips: {args.clips}\n")
        f.write(f"Tiles: {args.tiles}\n")
        f.write(f"Total configs: {len(rows)}\n\n")

        if rows:
            f.write("-"*80 + "\n")
            f.write("RAW BASELINE\n")
            f.write("-"*80 + "\n")
            f.write(f"UIQM:  {rows[0]['raw_uiqm_mean']:.4f} ± {rows[0]['raw_uiqm_std']:.4f}\n")
            f.write(f"UCIQE: {rows[0]['raw_uciqe_mean']:.4f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP 5 CONFIGS BY UIQM\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<5} {'Config':<20} {'UIQM':>10} {'ΔUIQM':>10} {'UCIQE':>10} {'ΔUCIQE':>10} {'CG':>8}\n")
        for i, r in enumerate(rows_by_uiqm[:5]):
            f.write(f"{i+1:<5} {r['config']:<20} {r['uiqm_mean']:>10.4f} {r['uiqm_delta_mean']:>+10.4f} "
                   f"{r['uciqe_mean']:>10.4f} {r['uciqe_delta_mean']:>+10.4f} {r['contrast_gain_mean']:>8.3f}\n")
        
        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("TOP 5 CONFIGS BY UCIQE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<5} {'Config':<20} {'UCIQE':>10} {'ΔUCIQE':>10} {'UIQM':>10} {'ΔUIQM':>10}\n")
        for i, r in enumerate(rows_by_uciqe[:5]):
            f.write(f"{i+1:<5} {r['config']:<20} {r['uciqe_mean']:>10.4f} {r['uciqe_delta_mean']:>+10.4f} "
                   f"{r['uiqm_mean']:>10.4f} {r['uiqm_delta_mean']:>+10.4f}\n")

        f.write("\n")
        f.write("-"*80 + "\n")
        f.write("FULL RESULTS (sorted by UIQM)\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Config':<20} {'UIQM':>8} {'±':>6} {'ΔUIQM':>8} {'UCIQE':>8} {'ΔUCIQE':>8} {'CG':>6} {'UIQM↑%':>7}\n")
        for r in rows_by_uiqm:
            f.write(f"{r['config']:<20} {r['uiqm_mean']:>8.4f} {r['uiqm_std']:>6.3f} {r['uiqm_delta_mean']:>+8.4f} "
                   f"{r['uciqe_mean']:>8.4f} {r['uciqe_delta_mean']:>+8.4f} {r['contrast_gain_mean']:>6.3f} "
                   f"{r['uiqm_improved_rate']:>6.1f}%\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        best_uiqm = rows_by_uiqm[0]
        best_uciqe = rows_by_uciqe[0]
        
        f.write(f"\nBest for UIQM (perceptual quality):\n")
        f.write(f"  Config: {best_uiqm['config']}\n")
        f.write(f"  UIQM: {best_uiqm['uiqm_mean']:.4f} (Δ={best_uiqm['uiqm_delta_mean']:+.4f})\n")
        f.write(f"  UCIQE: {best_uiqm['uciqe_mean']:.4f}\n")
        
        f.write(f"\nBest for UCIQE (contrast/colorfulness):\n")
        f.write(f"  Config: {best_uciqe['config']}\n")
        f.write(f"  UCIQE: {best_uciqe['uciqe_mean']:.4f} (Δ={best_uciqe['uciqe_delta_mean']:+.4f})\n")
        f.write(f"  UIQM: {best_uciqe['uiqm_mean']:.4f}\n")

        uiqm_ranks = {r['config']: i for i, r in enumerate(rows_by_uiqm)}
        uciqe_ranks = {r['config']: i for i, r in enumerate(rows_by_uciqe)}
        combined = [(r['config'], uiqm_ranks[r['config']] + uciqe_ranks[r['config']]) for r in rows]
        combined.sort(key=lambda x: x[1])
        best_balanced_cfg = combined[0][0]
        best_balanced = next(r for r in rows if r['config'] == best_balanced_cfg)
        
        f.write(f"\nBalanced (UIQM + UCIQE):\n")
        f.write(f"  Config: {best_balanced['config']}\n")
        f.write(f"  UIQM: {best_balanced['uiqm_mean']:.4f}, UCIQE: {best_balanced['uciqe_mean']:.4f}\n")


def write_latex_table(rows: List[Dict[str, Any]], out_path: str, num_samples: int) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    
    rows_sorted = sorted(rows, key=lambda r: (r['clip_limit'], r['tile_size']))

    best_uiqm = max(r['uiqm_mean'] for r in rows)
    best_uciqe = max(r['uciqe_mean'] for r in rows)
    
    clips = sorted(set(r['clip_limit'] for r in rows))
    tiles = sorted(set(r['tile_size'] for r in rows))
    
    with open(out_path, "w") as f:
        f.write("% CLAHE Grid Search Results\n")
        f.write(f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"% Samples: {num_samples}\n\n")

        f.write("% Table 1: UIQM by clip_limit and tile_size\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{CLAHE parameter grid search: UIQM on {num_samples} RUOD images. "
               f"Higher is better. Best result in bold.}}\n")
        f.write("\\label{tab:clahe_grid_uiqm}\n")
        
        cols = "c" + "c" * len(tiles)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\hline\n")
        f.write("Clip & " + " & ".join([f"${t}\\times{t}$" for t in tiles]) + " \\\\\n")
        f.write("\\hline\n")
        
        for clip in clips:
            row_data = [f"{clip:.1f}"]
            for tile in tiles:
                r = next((x for x in rows if x['clip_limit'] == clip and x['tile_size'] == tile), None)
                if r:
                    val = r['uiqm_mean']
                    delta = r['uiqm_delta_mean']
                    if abs(val - best_uiqm) < 0.0001:
                        row_data.append(f"\\textbf{{{val:.4f}}}")
                    else:
                        row_data.append(f"{val:.4f}")
                else:
                    row_data.append("--")
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        f.write("% Table 2: UCIQE by clip_limit and tile_size\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{CLAHE parameter grid search: UCIQE on {num_samples} RUOD images. "
               f"Higher is better.}}\n")
        f.write("\\label{tab:clahe_grid_uciqe}\n")
        
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\hline\n")
        f.write("Clip & " + " & ".join([f"${t}\\times{t}$" for t in tiles]) + " \\\\\n")
        f.write("\\hline\n")
        
        for clip in clips:
            row_data = [f"{clip:.1f}"]
            for tile in tiles:
                r = next((x for x in rows if x['clip_limit'] == clip and x['tile_size'] == tile), None)
                if r:
                    val = r['uciqe_mean']
                    if abs(val - best_uciqe) < 0.0001:
                        row_data.append(f"\\textbf{{{val:.4f}}}")
                    else:
                        row_data.append(f"{val:.4f}")
                else:
                    row_data.append("--")
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

        f.write("% Table 3: Combined results with deltas\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{CLAHE grid search results on {num_samples} images. "
               f"$\\Delta$ indicates change from raw baseline.}}\n")
        f.write("\\label{tab:clahe_grid_full}\n")
        f.write("\\begin{tabular}{cccccccc}\n")
        f.write("\\hline\n")
        f.write("Clip & Tile & UIQM & $\\Delta$UIQM & UCIQE & $\\Delta$UCIQE & Contrast & UIQM$\\uparrow$\\% \\\\\n")
        f.write("\\hline\n")
        
        for r in rows_sorted:
            uiqm_str = f"\\textbf{{{r['uiqm_mean']:.4f}}}" if abs(r['uiqm_mean'] - best_uiqm) < 0.0001 else f"{r['uiqm_mean']:.4f}"
            uciqe_str = f"\\textbf{{{r['uciqe_mean']:.4f}}}" if abs(r['uciqe_mean'] - best_uciqe) < 0.0001 else f"{r['uciqe_mean']:.4f}"
            
            f.write(f"{r['clip_limit']:.1f} & ${r['tile_size']}\\times{r['tile_size']}$ & "
                   f"{uiqm_str} & {r['uiqm_delta_mean']:+.4f} & "
                   f"{uciqe_str} & {r['uciqe_delta_mean']:+.4f} & "
                   f"{r['contrast_gain_mean']:.3f} & {r['uiqm_improved_rate']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standard CLAHE Grid Search")
    p.add_argument("--val_dir", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-list", type=str, default=None)
    p.add_argument("--save-sample-list", type=str, default=None)
    
    p.add_argument("--clips", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 6.0],
                   help="Clip limit values to test")
    p.add_argument("--tiles", type=int, nargs="+", default=[4, 8, 16, 32],
                   help="Tile sizes to test (NxN)")
    
    p.add_argument("--save-imgs", action="store_true", help="Save enhanced images for each config")
    p.add_argument("--out-dir", type=str, default="./results/clahe_grid_search")
    p.add_argument("--quiet", "-q", action="store_true")
    
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
    print(f"[INFO] Using {len(image_paths)} images")
    
    cfgs = get_grid_configs(clips=args.clips, tiles=args.tiles)
    print(f"[INFO] Grid search: {len(args.clips)} clips × {len(args.tiles)} tiles = {len(cfgs)} configs")
    print(f"[INFO] Clips: {args.clips}")
    print(f"[INFO] Tiles: {args.tiles}")
    
    if args.save_imgs:
        ensure_dir(args.out_dir)
        print(f"[INFO] Saving images to: {args.out_dir}/images/")

    results = []
    
    print("\n" + "="*80)
    print("GRID SEARCH PROGRESS")
    print("="*80)
    
    for i, cfg in enumerate(cfgs):
        clip = cfg['clip_limit']
        tile = cfg['tile_grid_size']
        cfg_name = cfg_to_name(clip, tile)
        
        if not args.quiet:
            print(f"[{i+1:02d}/{len(cfgs):02d}] {cfg_name}...", end=" ", flush=True)
        
        result = evaluate_single_config(
            image_paths=image_paths,
            val_root=args.val_dir,
            clip_limit=clip,
            tile_grid_size=tile,
            save_imgs=args.save_imgs,
            out_root=args.out_dir,
            verbose=False,
        )
        results.append(result)
        
        if not args.quiet:
            print(f"UIQM={result['uiqm_mean']:.4f} (Δ={result['uiqm_delta_mean']:+.4f}), "
                  f"UCIQE={result['uciqe_mean']:.4f} (Δ={result['uciqe_delta_mean']:+.4f})")
    
    results_by_uiqm = sorted(results, key=lambda r: r.get("uiqm_mean", 0), reverse=True)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    if results:
        print(f"\nRaw baseline: UIQM={results[0]['raw_uiqm_mean']:.4f}, UCIQE={results[0]['raw_uciqe_mean']:.4f}")
    
    print(f"\n{'Rank':<5} {'Config':<20} {'UIQM':>10} {'ΔUIQM':>10} {'UCIQE':>10} {'ΔUCIQE':>10}")
    print("-"*70)
    for i, r in enumerate(results_by_uiqm[:10]):
        print(f"{i+1:<5} {r['config']:<20} {r['uiqm_mean']:>10.4f} {r['uiqm_delta_mean']:>+10.4f} "
              f"{r['uciqe_mean']:>10.4f} {r['uciqe_delta_mean']:>+10.4f}")

    ensure_dir(args.out_dir)
    
    csv_path = os.path.join(args.out_dir, "grid_results.csv")
    write_csv(results_by_uiqm, csv_path)
    print(f"\n[INFO] Saved CSV: {csv_path}")

    summary_path = os.path.join(args.out_dir, "grid_results_summary.txt")
    write_summary(results, summary_path, args)
    print(f"[INFO] Saved summary: {summary_path}")

    latex_path = os.path.join(args.out_dir, "grid_table.tex")
    write_latex_table(results, latex_path, len(image_paths))
    print(f"[INFO] Saved LaTeX: {latex_path}")

    sample_list_path = os.path.join(args.out_dir, f"sample_list_{args.num_samples}_seed{args.seed}.txt")
    with open(sample_list_path, "w") as f:
        for p in image_paths:
            f.write(p + "\n")
    print(f"[INFO] Saved sample list: {sample_list_path}")
    
    best = results_by_uiqm[0]
    print("\n" + "="*80)
    print("BEST CONFIG (by UIQM)")
    print("="*80)
    print(f"  Config: {best['config']}")
    print(f"  UIQM:   {best['uiqm_mean']:.4f} ± {best['uiqm_std']:.4f} (Δ={best['uiqm_delta_mean']:+.4f})")
    print(f"  UCIQE:  {best['uciqe_mean']:.4f} ± {best['uciqe_std']:.4f} (Δ={best['uciqe_delta_mean']:+.4f})")
    print(f"  Contrast gain: {best['contrast_gain_mean']:.3f}")
    print(f"  UIQM improved: {best['uiqm_improved_rate']:.1f}% of images")
    
    if args.save_imgs:
        print(f"\n[INFO] Enhanced images saved to: {args.out_dir}/images/")
        print(f"       Each config folder can be used for YOLO mAP evaluation")


if __name__ == "__main__":
    main()