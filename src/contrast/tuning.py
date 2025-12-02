# src/contrast/tuning.py

import os
import cv2
import numpy as np

from .color_space import bgr_to_lab_luminance
from .stats import luminance_std, average_gradient
from .clahe import clahe_luminance


def list_images(folder):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".png"]
    paths = []
    for name in os.listdir(folder):
        lower = name.lower()
        if any(lower.endswith(e) for e in exts):
            paths.append(os.path.join(folder, name))
    return sorted(paths)


def evaluate_config_on_folder(folder, tile_size, clip_limit):
    img_paths = list_images(folder)
    if len(img_paths) == 0:
        print("No images found in:", folder)
        return None

    std_list = []
    grad_list = []

    for path in img_paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print("Warning: could not read", path)
            continue

        # 1) get luminance
        L, _ = bgr_to_lab_luminance(bgr)   # L in [0,1]

        # 2) apply CLAHE directly on L (force apply, no gating)
        L_enh = clahe_luminance(
            L,
            tile_size=tile_size,
            clip_limit=clip_limit,
            nbins=256,
        )

        # 3) compute stats on enhanced luminance
        std_val = luminance_std(L_enh)
        grad_val = average_gradient(L_enh)

        std_list.append(std_val)
        grad_list.append(grad_val)

    if len(std_list) == 0:
        return None

    avg_std = float(np.mean(std_list))
    avg_grad = float(np.mean(grad_list))
    return avg_std, avg_grad


def sweep_parameters(folder):
    tile_sizes = [6, 8, 12]
    clip_limits = [1.5, 2.0, 2.5, 3.0]

    print("Evaluating CLAHE configs on folder:", folder)
    print("----------------------------------------------------")
    print("tile_size | clip_limit | avg_std(L) | avg_grad(L)")
    print("----------------------------------------------------")

    results = []

    for t in tile_sizes:
        for c in clip_limits:
            scores = evaluate_config_on_folder(folder, t, c)
            if scores is None:
                continue
            avg_std, avg_grad = scores
            results.append((t, c, avg_std, avg_grad))
            print(f"{t:9d} | {c:9.2f} | {avg_std:9.4f} | {avg_grad:9.4f}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m src.contrast.tuning <image_folder>")
        print("Example: python -m src.contrast.tuning data/raw/")
    else:
        folder = sys.argv[1]
        sweep_parameters(folder)