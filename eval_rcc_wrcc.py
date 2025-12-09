import argparse
import glob
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.color_correction.rcc_wrcc import rcc_rgb, wrcc_rgb
from utils.quality_metrics import uiqm, uciqe, contrast_gain


def list_images(root_dir: str,
                exts: List[str] = (".jpg", ".jpeg", ".png", ".bmp")) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"),
                               recursive=True))
    return sorted(files)


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
        cfg_name = f"rcc_a{alpha:.1f}"
    else:  # wrcc
        cfg_name = f"wrcc_a{alpha:.1f}_w{window_size}_r{guided_radius}"

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
    uiqm_vals: List[float] = []
    uciqe_vals: List[float] = []
    cg_vals: List[float] = []

    if save_imgs and out_root is None:
        raise ValueError("out_root must be provided when save_imgs=True.")

    for path in image_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ===== 方法分支 =====
        if method == "raw":
            # 不做任何增强，直接用原图
            enhanced = rgb
        elif method == "rcc":
            enhanced = rcc_rgb(rgb, alpha=alpha)
        elif method == "wrcc":
            if window_size is None:
                raise ValueError("window_size must be set for WRCC.")
            enhanced = wrcc_rgb(
                rgb,
                alpha=alpha,
                window_size=window_size,
                guided_radius=guided_radius,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        # =====================

        uiqm_vals.append(uiqm(enhanced))
        uciqe_vals.append(uciqe(enhanced))
        # 对比度增益：raw 时就是对自己比，看你的实现一般会得到 1.0 或类似基准
        cg_vals.append(contrast_gain(rgb, enhanced))

        if save_imgs and method != "raw":
            # 原图一般不需要重复保存
            save_enhanced_image(
                enhanced_rgb=enhanced,
                orig_path=path,
                val_root=val_root,
                method=method,
                alpha=alpha,
                window_size=window_size,
                guided_radius=guided_radius,
                out_root=out_root,
            )

    if not uiqm_vals:
        return {"uiqm": 0.0, "uciqe": 0.0, "contrast_gain": 0.0}

    return {
        "uiqm": float(np.mean(uiqm_vals)),
        "uciqe": float(np.mean(uciqe_vals)),
        "contrast_gain": float(np.mean(cg_vals)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RCC / WRCC on underwater images.")
    parser.add_argument(
        "--val_dir",
        type=str,
        required=True,
        help="Path to validation image directory (e.g., RUOD val images).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="both",
        choices=["rcc", "wrcc", "both"],
        help="Which method(s) to evaluate.",
    )
    parser.add_argument(
        "--save_imgs",
        action="store_true",
        help="If set, save enhanced images for each setting.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/rcc_wrcc_images",
        help="Root directory to save enhanced images (used if --save_imgs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_dir = args.val_dir
    image_paths = list_images(val_dir)
    print(f"[INFO] Found {len(image_paths)} images in {val_dir}")

    if not image_paths:
        print("[ERROR] No images found. Check --val_dir.")
        return

    results: List[dict] = []

    # # ===== baseline =====
    # raw_metrics = evaluate_setting(
    #     image_paths=image_paths,
    #     val_root=val_dir,
    #     method="raw",
    #     alpha=1.0,       
    #     window_size=None,
    #     guided_radius=8,
    #     save_imgs=False,
    #     out_root=None,
    # )
    # cfg_raw = {"method": "raw"}
    # print("\nConfig:", cfg_raw)
    # print("Metrics:", raw_metrics)
    # results.append({**cfg_raw, **raw_metrics})
    # ============================================

    # 如需同时测 RCC，把下面这块取消注释即可
    # if args.methods in ("rcc", "both"):
    #     for alpha in [0.5, 1.0, 1.5, 2.0, 2.5]:
    #         metrics = evaluate_setting(
    #             image_paths=image_paths,
    #             val_root=val_dir,
    #             method="rcc",
    #             alpha=alpha,
    #             save_imgs=args.save_imgs,
    #             out_root=args.out_dir,
    #         )
    #         cfg = {"method": "rcc", "alpha": alpha}
    #         print("\nConfig:", cfg)
    #         print("Metrics:", metrics)
    #         results.append({**cfg, **metrics})

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

    results_sorted = sorted(results, key=lambda x: x["uiqm"], reverse=True)

    print("\n======================")
    print("Top configs by UIQM")
    print("======================")
    for r in results_sorted[:10]:
        print(r)


if __name__ == "__main__":
    main()
