"""
Underwater Image Fusion Pipeline

This script applies Ancuti-style multi-scale fusion to enhance underwater images.

Pipeline Options:
    1. Ancuti-style (Paper): Original → Gray-World WB → Input1
                                                       → Denoise → CLAHE → Input2
    
    2. RCC-first (Recommended): Original → RCC → Gray-World WB → Input1
                                                                → Denoise → CLAHE → Input2
    
    3. Parallel (Original 3-branch): Original → RCC → Input1
                                              → CLAHE → Input2
                                              → Denoise → Input3

Usage:
    # Recommended: RCC then White Balance (better red restoration)
    python run_fusion.py image.jpg -o results/ --color-correction rcc_wb
    
    # Paper's approach: White Balance only
    python run_fusion.py image.jpg -o results/ --color-correction gray_world
    
    # Your original: RCC only
    python run_fusion.py image.jpg -o results/ --color-correction rcc
    
    # Batch processing
    python run_fusion.py ./images/ -o results/ --batch --color-correction rcc_wb
"""

import os
import sys
import cv2
import numpy as np
import argparse
import glob
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path



# Add project root to path for metric imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# Import enhancement modules
try:
    from .enhance_wrappers import (
        apply_rcc_bgr,
        apply_clahe_bgr,
        apply_denoise_bgr,
        gray_world_white_balance,
        DenoiseConfig,
    )
    from .fusion.multiscale_fusion import (
        multi_scale_fusion_three,
        visualize_weights,
    )
except ImportError:
    # Fallback for running as standalone script
    sys.path.insert(0, str(Path(__file__).parent))
    
    from enhance_wrappers import (
        apply_rcc_bgr,
        apply_clahe_bgr,
        apply_denoise_bgr,
        gray_world_white_balance,
        DenoiseConfig,
    )
    from fusion.multiscale_fusion import (
        multi_scale_fusion_three,
        visualize_weights,
    )

# Import metrics
from metrics.compute_metrics import *


# ============================================================================
# Evaluation Classes and Functions
# ============================================================================

def save_comparison_image(original: np.ndarray, fused: np.ndarray, 
                         metrics_list: List[ImageMetrics], 
                         output_dir: str, base_name: str) -> str:
    """Save side-by-side comparison with metrics overlay."""
    h, w = original.shape[:2]
    
    # Resize if too large
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        fused = cv2.resize(fused, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    
    # Layout params
    top_margin = 60
    bottom_margin = 80
    side_margin = 40
    gap = 40
    border_thick = 2
    
    canvas_h = h + top_margin + bottom_margin
    canvas_w = side_margin * 2 + w * 2 + gap
    
    # Create canvas
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 245
    
    # Panel positions
    x1 = side_margin
    x2 = side_margin + w + gap
    y0 = top_margin
    
    # Paste images
    canvas[y0:y0+h, x1:x1+w] = original
    canvas[y0:y0+h, x2:x2+w] = fused
    
    # Add borders
    cv2.rectangle(canvas, (x1, y0), (x1+w, y0+h), (0, 0, 0), border_thick)
    cv2.rectangle(canvas, (x2, y0), (x2+w, y0+h), (0, 0, 0), border_thick)
    
    # Titles
    title_font = cv2.FONT_HERSHEY_SIMPLEX
    
    def put_centered(text, center_x, y, color, scale=1.0, thickness=2):
        (tw, th), _ = cv2.getTextSize(text, title_font, scale, thickness)
        cv2.putText(canvas, text, (center_x - tw // 2, y),
                   title_font, scale, color, thickness, cv2.LINE_AA)
    
    center1 = x1 + w // 2
    center2 = x2 + w // 2
    
    put_centered("ORIGINAL", center1, 40, (0, 0, 0))
    put_centered("ENHANCED (Fused)", center2, 40, (0, 90, 0))
    
    # Metrics overlay
    if metrics_list and len(metrics_list) >= 2:
        orig = metrics_list[0]
        final = metrics_list[-1]
        
        def rel_improve(new, old):
            if abs(old) < 1e-6:
                return 0.0
            return 100.0 * (new - old) / abs(old)
        
        uiqm_pct = rel_improve(final.uiqm, orig.uiqm)
        uciqe_pct = rel_improve(final.uciqe, orig.uciqe)
        cgain_pct = (final.contrast - 1.0) * 100.0
        
        metric_lines = [
            f"UIQM: {orig.uiqm:.2f} -> {final.uiqm:.2f}  ({uiqm_pct:+.1f}%)",
            f"UCIQE: {orig.uciqe:.2f} -> {final.uciqe:.2f}  ({uciqe_pct:+.1f}%)",
            f"Contrast: {final.contrast:.2f}x  ({cgain_pct:+.1f}%)",
        ]
        
        base_y = y0 + h + 30
        for i, line in enumerate(metric_lines):
            put_centered(line, center2, base_y + i * 20, (40, 40, 40), scale=0.6, thickness=1)
    
    # Save
    path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    cv2.imwrite(path, canvas)
    return path


@dataclass
class FusionConfig:
    """Configuration for the fusion pipeline."""
    # Pipeline mode: 'ancuti' (paper-style) or 'parallel' (3-branch)
    pipeline_mode: str = 'ancuti'
    
    # Color correction method:
    #   'gray_world': Gray-World WB only (paper's approach)
    #   'rcc': RCC/WRCC only (your original approach)
    #   'rcc_wb': RCC then Gray-World WB (recommended for best red restoration)
    color_correction: str = 'rcc_wb'
    
    # Gray-World WB parameters
    wb_lambda: float = 0.2  # Adjustment parameter [0, 0.5]
    
    # RCC/WRCC parameters
    use_wrcc: bool = True
    rcc_alpha: float = 0.5
    rcc_window_size: int = 9
    rcc_guided_radius: int = 2
    
    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    clahe_use_gating: bool = False
    
    # Denoise parameters
    denoise_gaussian_ksize: int = 5
    denoise_gaussian_sigma: float = 1.5
    denoise_median_ksize: int = 3
    denoise_bilateral_d: int = 7
    denoise_sharpen_strength: float = 0.3
    
    # Fusion parameters
    fusion_levels: int = 5
    lam_laplacian: float = 1.0
    lam_local_contrast: float = 1.0
    lam_saliency: float = 1.0
    lam_exposedness: float = 1.0
    exposedness_sigma: float = 0.25


@dataclass
class FusionResult:
    """Result of fusion pipeline."""
    fused: np.ndarray
    color_corrected: np.ndarray  # After RCC/WB/RCC+WB
    clahe: np.ndarray
    denoise: np.ndarray
    original: np.ndarray
    rcc_intermediate: Optional[np.ndarray] = None  # For rcc_wb mode
    info: Dict[str, Any] = field(default_factory=dict)


def run_fusion_pipeline(
    bgr_img: np.ndarray,
    config: Optional[FusionConfig] = None,
    verbose: bool = True,
) -> FusionResult:
    """
    Run the complete fusion pipeline on a single image.
    
    Args:
        bgr_img: Input BGR image
        config: FusionConfig with all parameters
        verbose: Print progress information
    
    Returns:
        FusionResult with all intermediate and final results
    """
    if config is None:
        config = FusionConfig()
    
    H, W = bgr_img.shape[:2]
    info = {
        'original_size': (H, W),
        'pipeline_mode': config.pipeline_mode,
        'color_correction': config.color_correction,
    }
    
    if config.pipeline_mode == 'ancuti':
        # ============================================================
        # ANCUTI-STYLE PIPELINE (Sequential approach)
        # Original → Color Correction → Input1
        #                            → Denoise → CLAHE → Input2
        # ============================================================
        if verbose:
            print(f"   [Mode: Ancuti-style sequential | Color: {config.color_correction}]")
        
        # Step 1: Color correction (multiple options)
        if verbose:
            print(f"   [1/3] Applying color correction: {config.color_correction}...")
        
        rcc_intermediate = None
        
        if config.color_correction == 'gray_world':
            # Paper's original: Gray-World WB only
            bgr_cc = gray_world_white_balance(bgr_img, lambda_param=config.wb_lambda)
            info['wb_applied'] = True
            info['rcc_applied'] = False
            
        elif config.color_correction == 'rcc':
            # Your original: RCC/WRCC only
            bgr_cc = apply_rcc_bgr(
                bgr_img,
                use_wrcc=config.use_wrcc,
                alpha=config.rcc_alpha,
                window_size=config.rcc_window_size,
                guided_radius=config.rcc_guided_radius,
            )
            info['rcc_applied'] = True
            info['rcc_type'] = 'wrcc' if config.use_wrcc else 'rcc'
            info['wb_applied'] = False
            
        elif config.color_correction == 'rcc_wb':
            # Recommended: RCC first, then Gray-World WB
            # This combines red restoration (RCC) with overall color cast removal (WB)
            
            # Step 1a: Apply RCC to restore red channel
            rcc_intermediate = apply_rcc_bgr(
                bgr_img,
                use_wrcc=config.use_wrcc,
                alpha=config.rcc_alpha,
                window_size=config.rcc_window_size,
                guided_radius=config.rcc_guided_radius,
            )
            
            # Step 1b: Apply Gray-World WB to remove remaining color cast
            bgr_cc = gray_world_white_balance(rcc_intermediate, lambda_param=config.wb_lambda)
            
            info['rcc_applied'] = True
            info['rcc_type'] = 'wrcc' if config.use_wrcc else 'rcc'
            info['wb_applied'] = True
            info['rcc_alpha'] = config.rcc_alpha
            info['wb_lambda'] = config.wb_lambda
            
        else:
            raise ValueError(f"Unknown color correction method: {config.color_correction}")
        
        bgr_cc = cv2.resize(bgr_cc, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Input 1: Color corrected
        input1 = bgr_cc
        info['input1_type'] = 'color_corrected'
        
        # Step 2: Denoise the color-corrected image
        if verbose:
            print("   [2/3] Applying denoise → CLAHE on color-corrected...")
        bgr_denoised = apply_denoise_bgr(
            bgr_cc,  # Apply to color-corrected, not original!
            gaussian_ksize=config.denoise_gaussian_ksize,
            gaussian_sigma=config.denoise_gaussian_sigma,
            median_ksize=config.denoise_median_ksize,
            bilateral_d=config.denoise_bilateral_d,
            sharpen_strength=config.denoise_sharpen_strength,
        )
        
        # Step 3: CLAHE on denoised color-corrected
        bgr_clahe, clahe_info = apply_clahe_bgr(
            bgr_denoised,  # Apply to denoised color-corrected!
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
            use_gating=config.clahe_use_gating,
        )
        bgr_clahe = cv2.resize(bgr_clahe, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Input 2: Contrast enhanced (built on color-corrected + denoised)
        input2 = bgr_clahe
        info['input2_type'] = 'clahe_on_denoised_cc'
        info['clahe_gain'] = clahe_info.get('contrast_gain', 1.0)
        
        # Step 4: Two-input fusion (as in paper)
        if verbose:
            print("   [3/3] Applying multi-scale fusion (2 inputs)...")
        
        from fusion.multiscale_fusion import multi_scale_fusion_two
        fused = multi_scale_fusion_two(
            input1,
            input2,
            levels=config.fusion_levels,
            lam_laplacian=config.lam_laplacian,
            lam_local_contrast=config.lam_local_contrast,
            lam_saliency=config.lam_saliency,
            lam_exposedness=config.lam_exposedness,
            verbose=verbose,
        )
        
        return FusionResult(
            fused=fused,
            color_corrected=input1,  # Final color correction result
            clahe=input2,            # CLAHE on denoised CC
            denoise=bgr_denoised,
            original=bgr_img,
            rcc_intermediate=rcc_intermediate,  # Only for rcc_wb mode
            info=info,
        )
    
    else:
        # ============================================================
        # PARALLEL PIPELINE (Original 3-branch approach)
        # Original → RCC → Input1
        # Original → CLAHE → Input2
        # Original → Denoise → Input3
        # ============================================================
        if verbose:
            print("   [Mode: Parallel 3-branch pipeline]")
        
        # Step 1: RCC/WRCC - Red Channel Compensation
        if verbose:
            print("   [1/4] Applying RCC/WRCC...")
        bgr_rcc = apply_rcc_bgr(
            bgr_img,
            use_wrcc=config.use_wrcc,
            alpha=config.rcc_alpha,
            window_size=config.rcc_window_size,
            guided_radius=config.rcc_guided_radius,
        )
        bgr_rcc = cv2.resize(bgr_rcc, (W, H), interpolation=cv2.INTER_LINEAR)
        info['rcc_applied'] = True
        
        # Step 2: CLAHE - Contrast Enhancement
        if verbose:
            print("   [2/4] Applying CLAHE...")
        bgr_clahe, clahe_info = apply_clahe_bgr(
            bgr_img,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
            use_gating=config.clahe_use_gating,
        )
        bgr_clahe = cv2.resize(bgr_clahe, (W, H), interpolation=cv2.INTER_LINEAR)
        info['clahe_applied'] = clahe_info.get('applied', True)
        info['clahe_gain'] = clahe_info.get('contrast_gain', 1.0)
        
        # Step 3: Denoising
        if verbose:
            print("   [3/4] Applying denoising...")
        bgr_denoise = apply_denoise_bgr(
            bgr_img,
            gaussian_ksize=config.denoise_gaussian_ksize,
            gaussian_sigma=config.denoise_gaussian_sigma,
            median_ksize=config.denoise_median_ksize,
            bilateral_d=config.denoise_bilateral_d,
            sharpen_strength=config.denoise_sharpen_strength,
        )
        bgr_denoise = cv2.resize(bgr_denoise, (W, H), interpolation=cv2.INTER_LINEAR)
        info['denoise_applied'] = True
        
        # Step 4: Multi-scale Fusion
        if verbose:
            print("   [4/4] Applying multi-scale fusion...")
        fused = multi_scale_fusion_three(
            bgr_rcc,
            bgr_clahe,
            bgr_denoise,
            levels=config.fusion_levels,
            lam_laplacian=config.lam_laplacian,
            lam_local_contrast=config.lam_local_contrast,
            lam_saliency=config.lam_saliency,
            lam_exposedness=config.lam_exposedness,
            exposedness_sigma=config.exposedness_sigma,
            verbose=verbose,
        )
        
        return FusionResult(
            fused=fused,
            color_corrected=bgr_rcc,
            clahe=bgr_clahe,
            denoise=bgr_denoise,
            original=bgr_img,
            info=info,
        )


def create_visualization(
    result: FusionResult,
    save_path: Optional[str] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Create a visualization grid showing all pipeline stages.
    
    Args:
        result: FusionResult from pipeline
        save_path: Optional path to save visualization
        show: Whether to display the visualization
    
    Returns:
        Visualization image (BGR)
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Determine layout based on whether we have RCC intermediate
    if result.rcc_intermediate is not None:
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert BGR to RGB for display
    def to_rgb(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    if result.rcc_intermediate is not None:
        # Extended layout for RCC→WB pipeline
        axes[0, 0].imshow(to_rgb(result.original))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(to_rgb(result.rcc_intermediate))
        axes[0, 1].set_title('After RCC')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(to_rgb(result.color_corrected))
        axes[0, 2].set_title('After RCC→WB')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(to_rgb(result.denoise))
        axes[0, 3].set_title('Denoised')
        axes[0, 3].axis('off')
        
        axes[1, 0].imshow(to_rgb(result.clahe))
        axes[1, 0].set_title(f'CLAHE (gain: {result.info.get("clahe_gain", 1.0):.2f}x)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(to_rgb(result.fused))
        axes[1, 1].set_title('Fused Result', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Side-by-side comparison
        h = min(result.original.shape[0], result.fused.shape[0])
        w = min(result.original.shape[1], result.fused.shape[1])
        orig_resized = cv2.resize(result.original, (w, h))
        fused_resized = cv2.resize(result.fused, (w, h))
        comparison = np.concatenate([orig_resized, fused_resized], axis=1)
        axes[1, 2].imshow(to_rgb(comparison))
        axes[1, 2].set_title('Original vs Fused')
        axes[1, 2].axis('off')
        
        # Add pipeline info
        pipeline_text = f"Pipeline: {result.info.get('color_correction', 'N/A')}\n"
        if result.info.get('rcc_applied'):
            pipeline_text += f"RCC α={result.info.get('rcc_alpha', 'N/A')}\n"
        if result.info.get('wb_applied'):
            pipeline_text += f"WB λ={result.info.get('wb_lambda', 'N/A')}"
        axes[1, 3].text(0.5, 0.5, pipeline_text, 
                       ha='center', va='center', fontsize=10,
                       transform=axes[1, 3].transAxes)
        axes[1, 3].axis('off')
        
    else:
        # Standard layout
        axes[0, 0].imshow(to_rgb(result.original))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(to_rgb(result.color_corrected))
        cc_method = result.info.get('color_correction', 'Color Corrected')
        axes[0, 1].set_title(f'{cc_method.upper()}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(to_rgb(result.clahe))
        axes[0, 2].set_title(f'CLAHE (gain: {result.info.get("clahe_gain", 1.0):.2f}x)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(to_rgb(result.denoise))
        axes[1, 0].set_title('Denoised')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(to_rgb(result.fused))
        axes[1, 1].set_title('Fused Result', fontweight='bold')
        axes[1, 1].axis('off')
        
        # Side-by-side comparison
        h = min(result.original.shape[0], result.fused.shape[0])
        w = min(result.original.shape[1], result.fused.shape[1])
        orig_resized = cv2.resize(result.original, (w, h))
        fused_resized = cv2.resize(result.fused, (w, h))
        comparison = np.concatenate([orig_resized, fused_resized], axis=1)
        axes[1, 2].imshow(to_rgb(comparison))
        axes[1, 2].set_title('Original vs Fused')
        axes[1, 2].axis('off')
    
    plt.suptitle('Underwater Image Fusion Pipeline', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved visualization to: {save_path}")
    
    if show:
        plt.show()
    
    # Convert figure to image
    fig.canvas.draw()
    try:
        # New matplotlib API
        vis_img = np.asarray(fig.canvas.buffer_rgba())
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGBA2BGR)
    except AttributeError:
        # Fallback for older matplotlib
        vis_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_img = vis_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    return vis_img


def process_single_image(
    input_path: str,
    output_dir: str,
    config: Optional[FusionConfig] = None,
    save_intermediates: bool = False,
    visualize: bool = False,
    evaluate: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Process a single image through the fusion pipeline with evaluation.
    
    Args:
        input_path: Path to input image
        output_dir: Directory for output files
        config: FusionConfig
        save_intermediates: Save RCC, CLAHE, denoise results
        visualize: Create and save visualization
        evaluate: Compute and save metrics
        verbose: Print progress
    
    Returns:
        Dictionary with processing info and metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[ERROR] Could not read: {input_path}")
        return {'success': False, 'path': input_path}
    
    base = os.path.splitext(os.path.basename(input_path))[0]
    
    if verbose:
        print(f"\nProcessing: {input_path}")
        print(f"   Size: {bgr.shape[1]}x{bgr.shape[0]}")
    
    # Run pipeline
    result = run_fusion_pipeline(bgr, config, verbose=verbose)
    
    # Save fused result
    fused_path = os.path.join(output_dir, f"{base}_fused.jpg")
    cv2.imwrite(fused_path, result.fused)
    if verbose:
        print(f"   Saved fused result to: {fused_path}")
    
    # Save original for reference
    cv2.imwrite(os.path.join(output_dir, f"{base}_original.jpg"), result.original)
    
    # Save intermediates if requested
    if save_intermediates:
        cv2.imwrite(os.path.join(output_dir, f"{base}_color_corrected.jpg"), result.color_corrected)
        cv2.imwrite(os.path.join(output_dir, f"{base}_clahe.jpg"), result.clahe)
        cv2.imwrite(os.path.join(output_dir, f"{base}_denoise.jpg"), result.denoise)
        if result.rcc_intermediate is not None:
            cv2.imwrite(os.path.join(output_dir, f"{base}_rcc_only.jpg"), result.rcc_intermediate)
        if verbose:
            print(f"   Saved intermediate results")
    
    # Evaluate metrics
    metrics_list = []
    if evaluate:
        if verbose:
            print("   Computing metrics...")
        
        metrics_list = [
            compute_metrics(result.original, result.original, "Original"),
        ]
        
        # Add RCC intermediate if available
        if result.rcc_intermediate is not None:
            metrics_list.append(
                compute_metrics(result.rcc_intermediate, result.original, "After RCC")
            )
        
        metrics_list.extend([
            compute_metrics(result.color_corrected, result.original, "Color Corrected"),
            compute_metrics(result.denoise, result.original, "Denoised"),
            compute_metrics(result.clahe, result.original, "CLAHE"),
            compute_metrics(result.fused, result.original, "Fused"),
        ])
        
        # Print metrics table
        if verbose:
            print_metrics_table(metrics_list, base)
        
        # Save metrics CSV
        csv_path = os.path.join(output_dir, f"{base}_metrics.csv")
        save_metrics_csv(metrics_list, csv_path)
        if verbose:
            print(f"   Saved metrics to: {csv_path}")
        
        # Save metrics plot
        plot_path = save_metrics_plot(metrics_list, output_dir, base)
        if plot_path and verbose:
            print(f"   Saved metrics plot to: {plot_path}")
        
        # Save comparison image with metrics overlay
        comparison_path = save_comparison_image(
            result.original, result.fused, metrics_list, output_dir, base
        )
        if verbose:
            print(f"   Saved comparison to: {comparison_path}")
    
    # Create visualization if requested
    if visualize:
        vis_path = os.path.join(output_dir, f"{base}_visualization.png")
        create_visualization(result, save_path=vis_path)
    
    # Build return dict
    ret = {
        'success': True,
        'path': input_path,
        'output_path': fused_path,
        **result.info,
    }
    
    # Add metrics to return dict
    if metrics_list:
        orig_m = metrics_list[0]
        fused_m = metrics_list[-1]
        ret['uiqm_original'] = orig_m.uiqm
        ret['uiqm_fused'] = fused_m.uiqm
        ret['uiqm_delta'] = fused_m.uiqm - orig_m.uiqm
        ret['uciqe_original'] = orig_m.uciqe
        ret['uciqe_fused'] = fused_m.uciqe
        ret['uciqe_delta'] = fused_m.uciqe - orig_m.uciqe
        ret['contrast_gain'] = fused_m.contrast
    
    return ret


def collect_image_paths(samples: List[str]) -> List[str]:
    """Collect image paths from files, directories, and glob patterns."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    paths = []
    
    for s in samples:
        s = os.path.expanduser(s)
        
        # Directory
        if os.path.isdir(s):
            for root, _, files in os.walk(s):
                for f in files:
                    if f.lower().endswith(exts):
                        paths.append(os.path.join(root, f))
            continue
        
        # Glob pattern
        if any(ch in s for ch in ['*', '?', '[']):
            for p in glob.glob(s):
                if os.path.isfile(p) and p.lower().endswith(exts):
                    paths.append(p)
            continue
        
        # Single file
        if os.path.isfile(s) and s.lower().endswith(exts):
            paths.append(s)
    
    return sorted(list(set(paths)))


def run_batch_processing(
    samples: List[str],
    output_dir: str,
    config: Optional[FusionConfig] = None,
    save_intermediates: bool = False,
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process multiple images and generate a report.
    
    Args:
        samples: List of paths/directories/patterns
        output_dir: Output directory
        config: FusionConfig
        save_intermediates: Save intermediate results
        max_images: Maximum number of images to process
        verbose: Print progress
    
    Returns:
        List of result dictionaries
    """
    paths = collect_image_paths(samples)
    if not paths:
        print("[ERROR] No valid image files found.")
        return []
    
    if max_images and max_images < len(paths):
        paths = paths[:max_images]
    
    os.makedirs(output_dir, exist_ok=True)
    fused_dir = os.path.join(output_dir, 'fused')
    os.makedirs(fused_dir, exist_ok=True)
    
    if save_intermediates:
        for subdir in ['color_corrected', 'clahe', 'denoise']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        if config and config.color_correction == 'rcc_wb':
            os.makedirs(os.path.join(output_dir, 'rcc_only'), exist_ok=True)
    
    print("=" * 60)
    print(f"Batch Processing: {len(paths)} images")
    print(f"Output directory: {output_dir}")
    if config:
        print(f"Color correction: {config.color_correction}")
    print("=" * 60)
    
    results = []
    iterator = tqdm(paths, desc="Processing", unit="img") if len(paths) > 1 else paths
    
    for path in iterator:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            results.append({'success': False, 'path': path})
            continue
        
        base = os.path.splitext(os.path.basename(path))[0]
        
        try:
            result = run_fusion_pipeline(bgr, config, verbose=False)
            
            # Save fused
            fused_path = os.path.join(fused_dir, f"{base}_fused.jpg")
            cv2.imwrite(fused_path, result.fused)
            
            # Save intermediates
            if save_intermediates:
                cv2.imwrite(os.path.join(output_dir, 'color_corrected', f"{base}_cc.jpg"), 
                           result.color_corrected)
                cv2.imwrite(os.path.join(output_dir, 'clahe', f"{base}_clahe.jpg"), 
                           result.clahe)
                cv2.imwrite(os.path.join(output_dir, 'denoise', f"{base}_denoise.jpg"), 
                           result.denoise)
                if result.rcc_intermediate is not None:
                    cv2.imwrite(os.path.join(output_dir, 'rcc_only', f"{base}_rcc.jpg"), 
                               result.rcc_intermediate)
            
            results.append({
                'success': True,
                'path': path,
                'output_path': fused_path,
                **result.info,
            })
            
        except Exception as e:
            results.append({'success': False, 'path': path, 'error': str(e)})
    
    # Write CSV report
    csv_path = os.path.join(output_dir, 'report.csv')
    fieldnames = ['path', 'success', 'color_correction', 'rcc_applied', 'wb_applied', 'clahe_gain']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    success_count = sum(1 for r in results if r.get('success', False))
    print("=" * 60)
    print(f"Completed: {success_count}/{len(results)} images")
    print(f"Report saved to: {csv_path}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Underwater Image Fusion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: RCC then White Balance
  python run_fusion.py image.jpg -o results/ --color-correction rcc_wb
  
  # Paper's approach: White Balance only
  python run_fusion.py image.jpg -o results/ --color-correction gray_world
  
  # Your original: RCC only
  python run_fusion.py image.jpg -o results/ --color-correction rcc
  
  # Batch processing with visualization
  python run_fusion.py ./images/ -o results/ --batch --color-correction rcc_wb -v
  
  # Custom parameters
  python run_fusion.py image.jpg -o results/ --color-correction rcc_wb \\
      --rcc-alpha 0.5 --wb-lambda 0.2 --clahe-clip 2.5
        """
    )
    
    # Input/Output
    parser.add_argument('input', nargs='+',
                        help='Input image(s), directory, or glob pattern')
    parser.add_argument('--output', '-o', default='fusion_output',
                        help='Output directory')
    
    # Mode
    parser.add_argument('--batch', action='store_true',
                        help='Batch processing mode')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Create visualization')
    parser.add_argument('--save-intermediates', '-s', action='store_true',
                        help='Save intermediate results')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum images to process in batch mode')
    
    # Pipeline mode
    parser.add_argument('--mode', choices=['ancuti', 'parallel'], default='ancuti',
                        help='Pipeline mode: "ancuti" (paper-style 2-input) or "parallel" (3-branch)')
    
    # Color correction method
    parser.add_argument('--color-correction', choices=['gray_world', 'rcc', 'rcc_wb'], 
                        default='rcc_wb',
                        help='Color correction: gray_world (paper), rcc (your original), rcc_wb (recommended)')
    
    # Gray-World WB parameters
    parser.add_argument('--wb-lambda', type=float, default=0.2,
                        help='Gray-World lambda parameter [0, 0.5] (default: 0.2)')
    
    # RCC parameters
    parser.add_argument('--use-rcc', action='store_true',
                        help='Use basic RCC instead of WRCC')
    parser.add_argument('--rcc-alpha', type=float, default=0.5,
                        help='RCC alpha parameter (default: 0.5)')
    parser.add_argument('--rcc-window-size', type=int, default=9,
                        help='WRCC window size (default: 9)')
    parser.add_argument('--rcc-guided-radius', type=int, default=2,
                        help='WRCC guided filter radius (default: 2)')
    
    # CLAHE parameters
    parser.add_argument('--clahe-clip', type=float, default=2.0,
                        help='CLAHE clip limit (default: 2.0)')
    parser.add_argument('--clahe-gating', action='store_true',
                        help='Enable CLAHE gating for high-contrast images')
    
    # Denoise parameters
    parser.add_argument('--denoise-bilateral-d', type=int, default=7,
                        help='Bilateral filter diameter (0 to skip)')
    parser.add_argument('--denoise-sharpen', type=float, default=0.3,
                        help='Sharpening strength (default: 0.3)')
    
    # Fusion parameters
    parser.add_argument('--fusion-levels', type=int, default=5,
                        help='Pyramid levels (default: 5)')
    parser.add_argument('--lam-laplacian', type=float, default=1.0,
                        help='Laplacian contrast weight exponent')
    parser.add_argument('--lam-local-contrast', type=float, default=1.0,
                        help='Local contrast weight exponent')
    parser.add_argument('--lam-saliency', type=float, default=1.0,
                        help='Saliency weight exponent')
    parser.add_argument('--lam-exposedness', type=float, default=1.0,
                        help='Exposedness weight exponent')
    
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--no-eval', action='store_true',
                        help='Skip metric evaluation')
    
    args = parser.parse_args()
    
    # Build config
    config = FusionConfig(
        pipeline_mode=args.mode,
        color_correction=args.color_correction,
        wb_lambda=args.wb_lambda,
        use_wrcc=not args.use_rcc,
        rcc_alpha=args.rcc_alpha,
        rcc_window_size=args.rcc_window_size,
        rcc_guided_radius=args.rcc_guided_radius,
        clahe_clip_limit=args.clahe_clip,
        clahe_use_gating=args.clahe_gating,
        denoise_bilateral_d=args.denoise_bilateral_d,
        denoise_sharpen_strength=args.denoise_sharpen,
        fusion_levels=args.fusion_levels,
        lam_laplacian=args.lam_laplacian,
        lam_local_contrast=args.lam_local_contrast,
        lam_saliency=args.lam_saliency,
        lam_exposedness=args.lam_exposedness,
    )
    
    verbose = not args.quiet
    
    if args.batch:
        run_batch_processing(
            samples=args.input,
            output_dir=args.output,
            config=config,
            save_intermediates=args.save_intermediates,
            max_images=args.max_images,
            verbose=verbose,
        )
    else:
        # Single image or small set
        paths = collect_image_paths(args.input)
        if not paths:
            print("[ERROR] No valid image files found.")
            return
        
        for path in paths:
            process_single_image(
                input_path=path,
                output_dir=args.output,
                config=config,
                save_intermediates=args.save_intermediates,
                visualize=args.visualize,
                evaluate=not args.no_eval,
                verbose=verbose,
            )


if __name__ == '__main__':
    main()