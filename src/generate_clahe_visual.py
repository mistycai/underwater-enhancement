# generate_pipeline_visuals.py
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src if running from root
sys.path.append(os.getcwd())

try:
    from contrast.color_space import bgr_to_lab_luminance, lab_luminance_to_bgr
    from contrast.stats import luminance_std, average_gradient
    from contrast.clahe import clahe_luminance
    from contrast.gating import should_apply_clahe
except ImportError:
    print("Error: Could not import 'src.contrast'. Make sure you run this script from the project root directory.")
    sys.exit(1)

# --- Configuration Constants ---
OUTPUT_DIR = "pipeline_visuals"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_img(name, img_bgr):
    path = os.path.join(OUTPUT_DIR, name)
    cv2.imwrite(path, img_bgr)
    print(f"Saved: {path}")

def save_plot(name, fig):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

def draw_grid(img, tile_size, color=(0, 255, 0), thickness=2):
    """Draws the CLAHE tiling grid on an image."""
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
    H, W = vis.shape[:2]
    # Vertical lines
    for x in range(0, W, tile_size):
        cv2.line(vis, (x, 0), (x, H), color, thickness)
    # Horizontal lines
    for y in range(0, H, tile_size):
        cv2.line(vis, (0, y), (W, y), color, thickness)
    return vis

def generate_histogram_viz(tile_pixels, clip_limit, nbins=256):
    """
    Generates the 'Step B: Histogram Clipping' visualization.
    Shows the original histogram, the cut-off, and the redistribution logic.
    """
    hist = np.bincount(tile_pixels.ravel(), minlength=nbins)[:nbins]
    n_pixels = tile_pixels.size
    
    # Calculate actual clip limit in pixel counts (same logic as your clahe.py)
    clip_val = clip_limit * (n_pixels / nbins)
    
    # Calculate clipped and excess
    hist_clipped = np.minimum(hist, clip_val)
    excess = np.sum(np.maximum(hist - clip_val, 0))
    # We don't actually redistribute in the plot to keep the "clipping" visual clear,
    # but we show the limit line.
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # 1. Original Histogram (faded gray)
    ax.bar(range(nbins), hist, width=1.0, color='lightgray', label='Original Peaks', alpha=0.6)
    
    # 2. Clipped Histogram (blue)
    ax.bar(range(nbins), hist_clipped, width=1.0, color='#1f77b4', label='Kept Region')
    
    # 3. Clip Limit Line (red dashed)
    ax.axhline(y=clip_val, color='red', linestyle='--', linewidth=2, label=f'Clip Limit ({clip_limit:.1f})')
    
    ax.set_title("Step B: Histogram Clipping")
    ax.set_xlim(0, 255)
    ax.legend()
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Count")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Generate visual assets for Detection-Aware CLAHE pipeline.")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--tile-size", type=int, default=60, help="Tile size in pixels (default: 60)")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="Clip limit (default: 2.0)")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)
    
    # --- 1. Load Input ---
    print(f"Processing: {args.input}")
    bgr = cv2.imread(args.input)
    if bgr is None:
        print("Error: Could not load image.")
        sys.exit(1)
        
    save_img("00_Input_Image.png", bgr)

    # --- Phase 1: Color Space Decomposition ---
    print("--- Phase 1: Decomposition ---")
    L_norm, lab = bgr_to_lab_luminance(bgr)  #
    
    L_uint8 = (L_norm * 255).astype(np.uint8)
    
    # Visualize channels
    # a/b channels usually look low contrast, so we apply a colormap for visualization
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    a_vis = cv2.applyColorMap(a_channel, cv2.COLORMAP_JET)
    b_vis = cv2.applyColorMap(b_channel, cv2.COLORMAP_CIVIDIS)
    
    save_img("01_Phase1_L_Channel.png", L_uint8)
    save_img("01_Phase1_a_Channel.png", a_vis)
    save_img("01_Phase1_b_Channel.png", b_vis)

    # --- Phase 2: Gating Stats ---
    print("--- Phase 2: Gating ---")
    sigma_L = luminance_std(L_norm)    #
    ag_L = average_gradient(L_norm)    #
    
    # Visualize Decision
    gating_viz = cv2.cvtColor(L_uint8, cv2.COLOR_GRAY2BGR)
    
    # Draw a stats box overlay
    h, w = gating_viz.shape[:2]
    box_w, box_h = 350, 130
    cv2.rectangle(gating_viz, (10, 10), (10 + box_w, 10 + box_h), (255, 255, 255), -1)
    cv2.rectangle(gating_viz, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(gating_viz, f"Sigma_L: {sigma_L:.4f}", (30, 40), font, 0.7, (0,0,0), 2)
    cv2.putText(gating_viz, f"AG_L:    {ag_L:.4f}",    (30, 70), font, 0.7, (0,0,0), 2)
    
    # Logic from gating.py
    # Defaults: sigma_thresh=0.18, grad_thresh=0.06
    do_apply = should_apply_clahe(L_norm, sigma_thresh=0.18, grad_thresh=0.06)
    
    color = (0, 180, 0) if do_apply else (0, 0, 255) # Green for YES, Red for SKIP
    text = "DECISION: APPLY" if do_apply else "DECISION: SKIP"
    cv2.putText(gating_viz, text, (30, 110), font, 0.8, color, 2)
    
    save_img("02_Phase2_Gating_Logic.png", gating_viz)

    # --- Phase 3: CLAHE Internals ---
    print("--- Phase 3: CLAHE Core ---")
    
    # Step A: Tiling Grid
    # Draw grid on the L channel
    grid_img = draw_grid(L_uint8, args.tile_size, color=(255, 255, 255))
    save_img("03_Phase3_StepA_Tiling.png", grid_img)
    
    # Step B: Histogram Clipping (Center Tile)
    cy, cx = h // 2, w // 2
    t = args.tile_size
    # Handle edge cases if image is smaller than tile
    y0, x0 = max(0, cy - t//2), max(0, cx - t//2)
    y1, x1 = min(h, y0 + t), min(w, x0 + t)
    
    tile_pixels = L_uint8[y0:y1, x0:x1]
    
    if tile_pixels.size > 0:
        fig_hist = generate_histogram_viz(tile_pixels, args.clip_limit)
        save_plot("03_Phase3_StepB_Histogram.png", fig_hist)
    
    # Step C: Enhanced L
    # We force apply CLAHE here regardless of gating decision to show what *would* happen
    L_enhanced = clahe_luminance(
        L_norm, 
        tile_size=args.tile_size, 
        clip_limit=args.clip_limit
    ) #
    
    L_enhanced_uint8 = (L_enhanced * 255).astype(np.uint8)
    save_img("03_Phase3_Output_Enhanced_L.png", L_enhanced_uint8)

    # --- Phase 4: Reconstruction ---
    print("--- Phase 4: Reconstruction ---")
    
    final_bgr = lab_luminance_to_bgr(L_enhanced, lab) #
    save_img("04_Phase4_Final_Output.png", final_bgr)

    print("\n[Done] Check 'pipeline_visuals/' for results.")

if __name__ == "__main__":
    main()