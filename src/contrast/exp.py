import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

def compute_uiqm(img_bgr: np.ndarray) -> float:
    '''TODO: double-check correctness of this uiqm'''
    img = img_bgr.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # UICM (colorfulness under water)
    RG = R - G
    YB = (R + G) / 2.0 - B
    uicm = (
        -0.0268 * np.sqrt(np.mean(RG)**2 + np.mean(YB)**2)
        + 0.1586 * np.sqrt(np.std(RG)**2 + np.std(YB)**2)
    )

    # UISM (sharpness)
    sobel_weight = [0.114, 0.587, 0.299]  # BGR weights
    uism_vals: List[float] = []
    for c in range(3):
        ch = img[:, :, c]
        sobel_x = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        uism_vals.append(np.mean(mag))
    uism = sum(w * v for w, v in zip(sobel_weight, uism_vals))

    # UIConM (local contrast)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    blk = 8
    contr = []
    H, W = gray.shape
    for i in range(0, H - blk, blk):
        for j in range(0, W - blk, blk):
            block = gray[i:i+blk, j:j+blk]
            mn, mx = block.min(), block.max()
            if mx > mn:
                contr.append((mx - mn) / (mx + mn + 1e-10))
    if len(contr) == 0:
        uiconm = 0.0
    else:
        uiconm = np.mean(np.log(np.array(contr) + 1e-10))

    # final
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm


def compute_uciqe(img_bgr: np.ndarray) -> float:
    ''' TODO: double-check correctness of uciqe '''
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    L = lab[:, :, 0] / 255.0
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)

    L_sorted = np.sort(L.flatten())
    n = len(L_sorted)
    top = L_sorted[int(0.99 * n):]
    bottom = L_sorted[:int(0.01 * n)] if int(0.01 * n) > 0 else L_sorted[:1]
    con_l = top.mean() - bottom.mean()

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    mu_s = hsv[:, :, 1].mean() / 255.0

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s


def contrast_gain(original: np.ndarray, enhanced: np.ndarray) -> float:
    '''Std(gray) ratio between enhanced and original'''
    def get_std(x: np.ndarray) -> float:
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    return get_std(enhanced) / (get_std(original) + 1e-10)


#### CLAHE Experiment

def apply_clahe(bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    '''Apply CLAHE on L channel in Lab'''
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def run_experiment(image_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(image_path)

    print(f"Running CLAHE experiment on: {image_path}")

    # parameter sweeps
    clip_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    tile_values = [4, 8, 12, 16]

    results = []

    for clip in clip_values:
        for tile in tile_values:
            enh = apply_clahe(orig, clip, tile)

            uiqm_val = compute_uiqm(enh)
            uciqe_val = compute_uciqe(enh)
            contr = contrast_gain(orig, enh)

            results.append({
                "clip": clip,
                "tile": tile,
                "uiqm": uiqm_val,
                "uciqe": uciqe_val,
                "contrast": contr,
            })

            print(
                f"[clip={clip:.1f}, tile={tile}] "
                f"UIQM={uiqm_val:.3f}, UCIQE={uciqe_val:.3f}, Contrast={contr:.3f}"
            )
    ### save results
    csv_path = os.path.join(out_dir, "clahe_experiment_results.csv")
    with open(csv_path, "w") as f:
        f.write("clip,tile,UIQM,UCIQE,ContrastGain\n")
        for r in results:
            f.write(
                f"{r['clip']},{r['tile']},"
                f"{r['uiqm']:.6f},{r['uciqe']:.6f},{r['contrast']:.6f}\n"
            )
    print(f"Saved CSV: {csv_path}")

    # Plot: UIQM vs clipLimit for each tile size
    plt.figure(figsize=(8, 5))
    for tile in tile_values:
        x = [r["clip"] for r in results if r["tile"] == tile]
        y = [r["uiqm"] for r in results if r["tile"] == tile]
        plt.plot(x, y, marker="o", label=f"Tile {tile}")
    plt.xlabel("clipLimit")
    plt.ylabel("UIQM")
    plt.title("UIQM vs clipLimit for different tile sizes (CLAHE)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    uiqm_plot_path = os.path.join(out_dir, "uiqm_vs_clip_all_tiles.png")
    plt.savefig(uiqm_plot_path, dpi=150)
    print(f"Saved: {uiqm_plot_path}")
    plt.close()

    # Plot: Three metrics vs clipLimit for ONE tile size
    target_tile = 8
    clips = [r["clip"] for r in results if r["tile"] == target_tile]
    uiqm_list = [r["uiqm"] for r in results if r["tile"] == target_tile]
    uciqe_list = [r["uciqe"] for r in results if r["tile"] == target_tile]
    contrast_list = [r["contrast"] for r in results if r["tile"] == target_tile]

    plt.figure(figsize=(8, 5))
    plt.plot(clips, uiqm_list, marker="o", label="UIQM")
    plt.plot(clips, uciqe_list, marker="s", label="UCIQE")
    plt.plot(clips, contrast_list, marker="^", label="Contrast Gain")
    plt.xlabel("clipLimit")
    plt.ylabel("Metric value")
    plt.title(f"CLAHE metrics vs clipLimit (tileGridSize = {target_tile}×{target_tile})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    metrics_plot_path = os.path.join(out_dir, "metrics_vs_clip_tile8.png")
    plt.savefig(metrics_plot_path, dpi=150)
    print(f"Saved: {metrics_plot_path}")
    plt.close()

    # find best setting
    best = max(
        results,
        key=lambda r: (r["uiqm"] + r["uciqe"] + r["contrast"])
    )
    print("\nBEST SETTING FOUND (by UIQM + UCIQE + ContrastGain):")
    print(best)

    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 experiment_clahe_params.py <image> <out_dir>")
        sys.exit(1)

    run_experiment(sys.argv[1], sys.argv[2])