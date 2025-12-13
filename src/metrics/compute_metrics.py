import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
import csv

from .uiqm import compute_uiqm
from .uciqe import compute_uciqe
from .entropy import entropy
from .contrast_gain import contrast_gain
from .colorfulness import colorfulness
from .avg_gradient import avg_gradient


@dataclass
class ImageMetrics:
    """Metrics for a single image."""
    name: str
    uiqm: float = 0.0
    uciqe: float = 0.0
    contrast: float = 1.0
    entropy_val: float = 0.0
    colorfulness_val: float = 0.0
    avg_gradient_val: float = 0.0
    uiqm_components: Dict = field(default_factory=dict)
    uciqe_components: Dict = field(default_factory=dict)


def compute_metrics(img: np.ndarray, original: np.ndarray, name: str) -> ImageMetrics:
    """Compute all metrics for an image."""
    uiqm_val, uiqm_comp = compute_uiqm(img)
    uciqe_val, uciqe_comp = compute_uciqe(img)
    
    return ImageMetrics(
        name=name,
        uiqm=uiqm_val,
        uciqe=uciqe_val,
        contrast=contrast_gain(original, img),
        entropy_val=entropy(img),
        colorfulness_val=colorfulness(img),
        avg_gradient_val=avg_gradient(img),
        uiqm_components=uiqm_comp,
        uciqe_components=uciqe_comp,
    )


def print_metrics_table(metrics_list: List[ImageMetrics], image_name: str):
    """Print metrics in a formatted table."""
    print(f"\n{'='*95}")
    print(f"EVALUATION: {image_name}")
    print(f"{'='*95}")
    print(f"{'Stage':<25} {'UIQM':>10} {'UCIQE':>10} {'Contrast':>10} "
          f"{'Entropy':>10} {'Colorful':>10} {'AvgGrad':>10}")
    print(f"{'-'*95}")
    
    for m in metrics_list:
        print(f"{m.name:<25} {m.uiqm:>10.4f} {m.uciqe:>10.2f} "
              f"{m.contrast:>10.4f} {m.entropy_val:>10.4f} "
              f"{m.colorfulness_val:>10.2f} {m.avg_gradient_val:>10.2f}")
    
    print(f"{'-'*95}")
    
    # Print improvement summary
    if len(metrics_list) >= 2:
        orig, final = metrics_list[0], metrics_list[-1]
        print(f"\nIMPROVEMENT (Original → Fused):")
        print(f"   UIQM:       {orig.uiqm:.4f} → {final.uiqm:.4f}  (Δ = {final.uiqm - orig.uiqm:+.4f})")
        print(f"   UCIQE:      {orig.uciqe:.2f} → {final.uciqe:.2f}  (Δ = {final.uciqe - orig.uciqe:+.2f})")
        print(f"   Contrast:   {final.contrast:.4f}x")
        
        # Percentage improvements
        if abs(orig.uiqm) > 1e-6:
            uiqm_pct = 100 * (final.uiqm - orig.uiqm) / abs(orig.uiqm)
            print(f"   UIQM improvement: {uiqm_pct:+.1f}%")
        if abs(orig.uciqe) > 1e-6:
            uciqe_pct = 100 * (final.uciqe - orig.uciqe) / abs(orig.uciqe)
            print(f"   UCIQE improvement: {uciqe_pct:+.1f}%")


def save_metrics_csv(metrics_list: List[ImageMetrics], csv_path: str):
    """Save metrics to CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Stage', 'UIQM', 'UCIQE', 'Contrast_Gain', 
                        'Entropy', 'Colorfulness', 'Avg_Gradient'])
        for m in metrics_list:
            writer.writerow([m.name, f"{m.uiqm:.4f}", f"{m.uciqe:.2f}", 
                           f"{m.contrast:.4f}", f"{m.entropy_val:.4f}",
                           f"{m.colorfulness_val:.2f}", f"{m.avg_gradient_val:.2f}"])


def save_metrics_plot(metrics_list: List[ImageMetrics], output_dir: str, base_name: str):
    """Save metrics bar plot."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    stages = [m.name for m in metrics_list]
    colors = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f'Enhancement Pipeline Metrics: {base_name}', fontsize=14, fontweight='bold')
    
    metrics_data = [
        ('UIQM', [m.uiqm for m in metrics_list]),
        ('UCIQE', [m.uciqe for m in metrics_list]),
        ('Contrast Gain', [m.contrast for m in metrics_list]),
        ('Entropy', [m.entropy_val for m in metrics_list]),
        ('Colorfulness', [m.colorfulness_val for m in metrics_list]),
        ('Avg Gradient', [m.avg_gradient_val for m in metrics_list]),
    ]
    
    for idx, (title, values) in enumerate(metrics_data):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(stages, values, color=colors[:len(stages)], edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_name}_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path

