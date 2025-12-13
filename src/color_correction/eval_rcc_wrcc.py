import argparse
import glob
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np

from .rcc_wrcc import rcc_rgb, wrcc_rgb
from ..metrics.compute_metrics import compute_metrics


def list_images(
    root_dir: str,
    exts: List[str] = (".jpg", ".jpeg", ".png", ".bmp")
) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext.upper()}"), recursive=True))
    # unique + stable order
    return sorted(set(files))


def sample_images(
    image_paths: List[str],
    num_samples: Optional[int],
    seed: int = 42,
    sample_list: Optional[str] = None,
    save_sample_list: Optional[str] = None,
) -> List[str]:
    """
    Priority:
      1) If sample_list is provided and exists => read paths from file (most reproducible)
      2) Else if num_samples is None or >= len(image_paths) => use all
      3) Else => random sample with fixed seed
    """
    if sample_list:
        if not os.path.isfile(sample_list):
            raise FileNotFoundError(f"--sample-list not found: {sample_list}")
        with open(sample_list, "r") as f:
            chosen = [line.strip() for line in f if line.strip()]
        # Keep only paths that still exist (optional safety)
        chosen = [p for p in chosen if os.path.isfile(p)]
        if not chosen:
            raise RuntimeError(f"--sample-list is empty or files missing: {sample_list}")
        return chosen

    if num_samples is None or num_samples >= len(image_paths):
        chosen = image_paths
    else:
        rng = random.Random(seed)
        chosen = rng.sample(image_paths, num_samples)
        chosen = sorted(chosen)  # keep output stable

    if save_sample_list:
        os.makedirs(os.path.dirname(save_sample_list) or ".", exist_ok=True)
        with open(save_sample_list, "w") as f:
            for p in chosen:
                f.write(p + "\n")

    return chosen


def save_enhanced_image(
    enhanced_rgb: np.ndarray,
    orig_path: str,
    val_root: str,
    method: str,
    alpha: float,
    window_size: Optional[int],
    guided_radius: Optional[int],
    out_root: str,
) -> None:
    rel_path = os.path.relpath(orig_path, val_root)
    rel_dir = os.path.dirname(rel_path)
    base = os.path.splitext(os.path.basename(rel_path))[0]

    if method == "rcc":
        cfg_name = f"rcc_a{alpha:.2f}"
    else:  # wrcc
        cfg_name = f"wrcc_a{alpha:.2f}_w{window_size}_r{guided_radius}"

    out_dir = os.path.join(out_root, cfg_name, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, base + ".jpg")

    bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


def evaluate_setting(
    image_paths: List[str],
    val_root: str,
    method: str,
    alpha: float,
    window_size: int = None,
    guided_radius: int = 8,
    save_imgs: bool = False,
    out_root: Optional[str] = None,
) -> Dict[str, float]:
    """
    Returns mean metrics over all images for the given setting.
    Uses metrics.compute_metrics(img, original, name) => ImageMetrics
    """
    uiqm_vals: List[float] = []
    uciqe_vals: List[float] = []
    contrast_vals: List[float] = []
    entropy_vals: List[float] = []
    colorful_vals: List[float] = []
    avggrad_vals: List[float] = []

    uiqm_deltas: List[float] = []
    uciqe_deltas: List[float] = []
    contrast_deltas: List[float] = []

    if save_imgs and out_root is None:
        raise ValueError("out_root must be provided when save_imgs=True.")

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raw_m = compute_metrics(bgr, bgr, name="raw")

        if method == "raw":
            enhanced_rgb = rgb
        elif method == "rcc":
            enhanced_rgb = rcc_rgb(rgb, alpha=alpha)
        elif method == "wrcc":
            if window_size is None:
                raise ValueError("window_size must be set for WRCC.")
            enhanced_rgb = wrcc_rgb(
                rgb,
                alpha=alpha,
                window_size=window_size,
                guided_radius=guided_radius,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        enh_m = compute_metrics(enhanced_bgr, bgr, name=method)

        uiqm_vals.append(enh_m.uiqm)
        uciqe_vals.append(enh_m.uciqe)
        contrast_vals.append(enh_m.contrast)
        entropy_vals.append(enh_m.entropy_val)
        colorful_vals.append(enh_m.colorfulness_val)
        avggrad_vals.append(enh_m.avg_gradient_val)

        uiqm_deltas.append(enh_m.uiqm - raw_m.uiqm)
        uciqe_deltas.append(enh_m.uciqe - raw_m.uciqe)
        contrast_deltas.append(enh_m.contrast - 1.0)

        if save_imgs and method != "raw":
            save_enhanced_image(
                enhanced_rgb=enhanced_rgb,
                orig_path=path,
                val_root=val_root,
                method=method,
                alpha=alpha,
                window_size=window_size,
                guided_radius=guided_radius,
                out_root=out_root,
            )

    if not uiqm_vals:
        return {
            "uiqm": 0.0,
            "uciqe": 0.0,
            "contrast_gain": 0.0,
            "entropy": 0.0,
            "colorfulness": 0.0,
            "avg_gradient": 0.0,
            "uiqm_delta": 0.0,
            "uciqe_delta": 0.0,
            "contrast_delta": 0.0,
            "num_images_used": 0,
        }

    return {
        "uiqm": float(np.mean(uiqm_vals)),
        "uciqe": float(np.mean(uciqe_vals)),
        "contrast_gain": float(np.mean(contrast_vals)),
        "entropy": float(np.mean(entropy_vals)),
        "colorfulness": float(np.mean(colorful_vals)),
        "avg_gradient": float(np.mean(avggrad_vals)),
        "uiqm_delta": float(np.mean(uiqm_deltas)),
        "uciqe_delta": float(np.mean(uciqe_deltas)),
        "contrast_delta": float(np.mean(contrast_deltas)),
        "num_images_used": int(len(uiqm_vals)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RCC / WRCC on underwater images.")
    parser.add_argument("--val_dir", type=str, required=True)

    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["raw", "rcc", "wrcc", "both"],
    )

    parser.add_argument("--save_imgs", action="store_true")
    parser.add_argument("--out_dir", type=str, default="results/rcc_wrcc_images")

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample N images from val_dir (default: use all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --num-samples is set.",
    )

    # optional but super useful for fair ablations across different scripts
    parser.add_argument(
        "--sample-list",
        type=str,
        default=None,
        help="If provided, load image paths from this txt file (one path per line).",
    )
    parser.add_argument(
        "--save-sample-list",
        type=str,
        default=None,
        help="If provided, save the chosen sample paths to this txt file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_dir = args.val_dir
    all_paths = list_images(val_dir)
    print(f"[INFO] Found {len(all_paths)} images in {val_dir}")

    if not all_paths:
        print("[ERROR] No images found. Check --val_dir.")
        return

    image_paths = sample_images(
        image_paths=all_paths,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_list=args.sample_list,
        save_sample_list=args.save_sample_list,
    )
    print(f"[INFO] Using {len(image_paths)} images for evaluation.")
    if args.num_samples is not None:
        print(f"[INFO] Sampling mode: num_samples={args.num_samples}, seed={args.seed}")
    if args.sample_list:
        print(f"[INFO] Loaded sample list: {args.sample_list}")
    if args.save_sample_list:
        print(f"[INFO] Saved sample list to: {args.save_sample_list}")

    results: List[dict] = []

    # --- raw baseline ---
    if args.methods in ("raw", "both"):
        raw_metrics = evaluate_setting(
            image_paths=image_paths,
            val_root=val_dir,
            method="raw",
            alpha=1.0,
            window_size=None,
            guided_radius=8,
            save_imgs=False,
            out_root=None,
        )
        cfg_raw = {"method": "raw"}
        print("\nConfig:", cfg_raw)
        print("Metrics:", raw_metrics)
        results.append({**cfg_raw, **raw_metrics})

    # --- RCC sweep ---
    if args.methods in ("rcc", "both"):
        for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
            metrics = evaluate_setting(
                image_paths=image_paths,
                val_root=val_dir,
                method="rcc",
                alpha=alpha,
                save_imgs=args.save_imgs,
                out_root=args.out_dir,
            )
            cfg = {"method": "rcc", "alpha": alpha}
            print("\nConfig:", cfg)
            print("Metrics:", metrics)
            results.append({**cfg, **metrics})

    # --- WRCC sweep ---
    if args.methods in ("wrcc", "both"):
        for alpha in [0.5]:
            for w in [9]:
                for r in [2]:
                    metrics = evaluate_setting(
                        image_paths=image_paths,
                        val_root=val_dir,
                        method="wrcc",
                        alpha=alpha,
                        window_size=w,
                        guided_radius=r,
                        save_imgs=args.save_imgs,
                        out_root=args.out_dir,
                    )
                    cfg = {
                        "method": "wrcc",
                        "alpha": alpha,
                        "window_size": w,
                        "guided_radius": r,
                    }
                    print("\nConfig:", cfg)
                    print("Metrics:", metrics)
                    results.append({**cfg, **metrics})

    results_sorted = sorted(results, key=lambda x: x.get("uiqm", 0.0), reverse=True)

    print("\n======================")
    print("Top configs by UIQM")
    print("======================")
    for r in results_sorted[:10]:
        print(r)


if __name__ == "__main__":
    main()