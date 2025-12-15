import os
import sys
import cv2
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from uiqm import compute_uiqm
from uciqe import compute_uciqe
from entropy import entropy
from contrast_gain import contrast_gain
from colorfulness import colorfulness
from avg_gradient import avg_gradient
from src.contrast.pipeline import apply_clahe_to_bgr
from src.fusion import multi_scale_fusion



def apply_rcc(bgr: np.ndarray, alpha: float = 1.2) -> np.ndarray:
    img = bgr.astype(np.float64)
    B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mu_r, mu_g = np.mean(R), np.mean(G)
    if mu_r > 0:
        R_comp = np.clip(R * (mu_g / mu_r) ** alpha, 0, 255)
    else:
        R_comp = R
    return np.stack([B, G, R_comp], axis=2).astype(np.uint8)
    pass


def apply_clahe(bgr: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def apply_denoise(bgr: np.ndarray) -> np.ndarray:
    img = cv2.GaussianBlur(bgr, (3, 3), 0.8)
    img = cv2.bilateralFilter(img, 5, 15, 15)
    return img

@dataclass
class StageMetrics:
    name: str
    uiqm: float
    uciqe: float
    contrast_gain: float
    entropy: float
    colorfulness: float
    avg_gradient: float
    uiqm_components: Dict
    uciqe_components: Dict


def evaluate_stage(img: np.ndarray, original: np.ndarray, name: str) -> StageMetrics:
    uiqm_val, uiqm_comp = compute_uiqm(img)
    uciqe_val, uciqe_comp = compute_uciqe(img)
    return StageMetrics(
        name=name, uiqm=uiqm_val, uciqe=uciqe_val,
        contrast_gain=contrast_gain(original, img),
        entropy=entropy(img), colorfulness=colorfulness(img),
        avg_gradient=avg_gradient(img),
        uiqm_components=uiqm_comp, uciqe_components=uciqe_comp
    )


def print_results(results: List[StageMetrics], image_name: str):
    print(f"UNDERWATER IMAGE ENHANCEMENT EVALUATION: {image_name}")
    print(f"{'Stage':<15} {'UIQM':>10} {'UCIQE':>10} {'Contrast':>10} "
          f"{'Entropy':>10} {'Colorful':>12} {'AvgGrad':>10}")
    print("-"*95)
    
    for r in results:
        print(f"{r.name:<15} {r.uiqm:>10.4f} {r.uciqe:>10.2f} "
              f"{r.contrast_gain:>10.4f} {r.entropy:>10.4f} "
              f"{r.colorfulness:>12.2f} {r.avg_gradient:>10.2f}")
    
    print("-"*95)
    
    orig, final = results[0], results[-1]
    print(f"\nIMPROVEMENT (Original -> Fused):")
    print(f"   UIQM:  {orig.uiqm:.4f} → {final.uiqm:.4f}  (Δ = {final.uiqm - orig.uiqm:+.4f})")
    print(f"   UCIQE: {orig.uciqe:.2f} → {final.uciqe:.2f}  (Δ = {final.uciqe - orig.uciqe:+.2f})")
    print(f"   Contrast Gain: {final.contrast_gain:.4f}x")


def save_plots(results: List[StageMetrics], output_dir: str, base_name: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed")
        return
    
    stages = [r.name for r in results]
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Enhancement Pipeline Metrics: {base_name}', fontsize=14, fontweight='bold')
    
    metrics = [
        ('UIQM', [r.uiqm for r in results]),
        ('UCIQE', [r.uciqe for r in results]),
        ('Contrast Gain', [r.contrast_gain for r in results]),
        ('Entropy', [r.entropy for r in results]),
        ('Colorfulness', [r.colorfulness for r in results]),
        ('Avg Gradient', [r.avg_gradient for r in results]),
    ]
    
    for idx, (title, values) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(stages, values, color=colors[:len(stages)], edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_name}_metrics.png"), dpi=150, bbox_inches='tight')
    print(f"Saved: {base_name}_metrics.png")
    plt.close()


def save_comparison_image(original, fused, results, output_dir, base_name):
    h, w = original.shape[:2]

    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fused    = cv2.resize(fused,    (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    top_margin    = 60   
    bottom_margin = 80    
    side_margin   = 40
    gap           = 40
    border_thick  = 2

    canvas_h = h + top_margin + bottom_margin
    canvas_w = side_margin*2 + w*2 + gap

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245

    x1 = side_margin
    x2 = side_margin + w + gap
    y0 = top_margin

    canvas[y0:y0+h, x1:x1+w] = original
    canvas[y0:y0+h, x2:x2+w] = fused

    cv2.rectangle(canvas, (x1, y0), (x1+w, y0+h), (0, 0, 0), border_thick)
    cv2.rectangle(canvas, (x2, y0), (x2+w, y0+h), (0, 0, 0), border_thick)

    title_font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.0
    title_thick = 2

    def put_centered(text, center_x, y, color):
        (tw, th), _ = cv2.getTextSize(text, title_font, title_scale, title_thick)
        cv2.putText(canvas, text,
                    (center_x - tw // 2, y),
                    title_font, title_scale, color, title_thick, cv2.LINE_AA)

    center1 = x1 + w // 2
    center2 = x2 + w // 2

    put_centered("ORIGINAL", center1, 40, (0, 0, 0))
    put_centered("ENHANCED (Fused)", center2, 40, (0, 90, 0))

    if results is not None and len(results) >= 2:
        orig = results[0]
        fused_m = results[-1]

        def rel_improve(new, old):
            if abs(old) < 1e-6:
                return 0.0
            return 100.0 * (new - old) / abs(old)

        uiqm_pct  = rel_improve(fused_m.uiqm,  orig.uiqm)
        uciqe_pct = rel_improve(fused_m.uciqe, orig.uciqe)
        cgain_pct = (fused_m.contrast_gain - 1.0) * 100.0

        metric_lines = [
            f"UIQM: {orig.uiqm:.2f} → {fused_m.uiqm:.2f}  ({uiqm_pct:+.1f}%)",
            f"UCIQE: {orig.uciqe:.2f} → {fused_m.uciqe:.2f}  ({uciqe_pct:+.1f}%)",
            f"Contrast: {fused_m.contrast_gain:.2f}×  ({cgain_pct:+.1f}%)",
        ]

        base_y = y0 + h + 30
        metric_font  = cv2.FONT_HERSHEY_SIMPLEX
        metric_scale = 0.6
        metric_thick = 1
        metric_color = (40, 40, 40)

        for i, line in enumerate(metric_lines):
            (tw, th), _ = cv2.getTextSize(line, metric_font, metric_scale, metric_thick)
            cv2.putText(canvas, line,
                        (center2 - tw // 2, base_y + i * (th + 6)),
                        metric_font, metric_scale, metric_color, metric_thick, cv2.LINE_AA)

    path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    cv2.imwrite(path, canvas)
    print(f"Saved comparison: {path}")


def run_evaluation(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read: {input_path}")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    H, W = original.shape[:2]
    
    print(f"\nEvaluating: {input_path}")
    print(f"   Image size: {W}x{H}")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for root in [os.path.dirname(os.path.dirname(script_dir)), os.path.dirname(script_dir), script_dir]:
            if root not in sys.path:
                sys.path.insert(0, root)
        
        from src.enhance_wrappers import apply_rcc_bgr, apply_clahe_bgr, apply_denoise_pipeline
        print("   Using team enhancement modules")
        
        rcc = cv2.resize(apply_rcc_bgr(original, use_wrcc=True), (W, H))
        clahe, _ = apply_clahe_to_bgr(original)
        clahe = cv2.resize(clahe, (W, H))
        denoise = cv2.resize(apply_denoise_pipeline(original), (W, H))
        
    except ImportError:
        print("   Using built-in enhancements (except CLAHE)")
        rcc = apply_rcc(original)
        clahe = apply_clahe_to_bgr(original)
        denoise = apply_denoise(original)
    
    fused = multi_scale_fusion(rcc, clahe, denoise, levels=4, verbose=True)
    fused = cv2.resize(fused, (W, H))

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), original)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_rcc.jpg"), rcc)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_clahe.jpg"), clahe)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoise.jpg"), denoise)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_fused.jpg"), fused)
    

    print("   Computing metrics...")
    results = [
        evaluate_stage(original, original, "Original"),
        evaluate_stage(rcc, original, "RCC"),
        evaluate_stage(clahe, original, "CLAHE"),
        evaluate_stage(denoise, original, "Denoise"),
        evaluate_stage(fused, original, "Fused"),
    ]
    
    print_results(results, base_name)
    save_plots(results, output_dir, base_name)
    save_comparison_image(original, fused, results, output_dir, base_name)
    
    csv_path = os.path.join(output_dir, f"{base_name}_metrics.csv")
    with open(csv_path, 'w') as f:
        f.write("Stage,UIQM,UCIQE,Contrast_Gain,Entropy,Colorfulness,Avg_Gradient\n")
        for r in results:
            f.write(f"{r.name},{r.uiqm:.4f},{r.uciqe:.2f},{r.contrast_gain:.4f},"
                    f"{r.entropy:.4f},{r.colorfulness:.2f},{r.avg_gradient:.2f}\n")
    
    print(f"\nDone! Results in: {output_dir}/")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/demo"
    run_evaluation(input_path, output_dir)