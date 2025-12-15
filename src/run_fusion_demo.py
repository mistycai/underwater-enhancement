# generate demo images showing fusion pipeline stages
# output: original, rcc, clahe, denoise, fused (side-by-side grid)

import os
import sys
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from run_fusion import FusionConfig, run_fusion_pipeline
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_fusion import FusionConfig, run_fusion_pipeline


def create_demo_grid(result, title_height=40, gap=10, max_width=1800):
    # create 2x3 grid: [original, rcc, clahe] / [denoise, fused, comparison]
    images = [
        ('original', result.original),
        ('rcc', result.rcc),
        ('clahe', result.clahe),
        ('denoise', result.denoise),
        ('fused', result.fused),
    ]
    
    h, w = result.original.shape[:2]
    
    # calculate cell size to fit max_width
    n_cols = 3
    cell_w = (max_width - gap * (n_cols + 1)) // n_cols
    scale = min(1.0, cell_w / w)
    cell_w = int(w * scale)
    cell_h = int(h * scale)
    
    # resize all images
    resized = []
    for name, img in images:
        r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        resized.append((name, r))
    
    # create original vs fused comparison
    orig_half = cv2.resize(result.original, (cell_w // 2, cell_h), interpolation=cv2.INTER_AREA)
    fused_half = cv2.resize(result.fused, (cell_w // 2, cell_h), interpolation=cv2.INTER_AREA)
    comparison = np.hstack([orig_half, fused_half])
    # add dividing line
    comparison[:, cell_w // 2 - 1 : cell_w // 2 + 1] = [0, 255, 255]
    resized.append(('orig | fused', comparison))
    
    # canvas size
    n_rows = 2
    canvas_w = gap + n_cols * (cell_w + gap)
    canvas_h = gap + n_rows * (cell_h + title_height + gap)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 40  # dark gray bg
    
    # place images
    for idx, (name, img) in enumerate(resized):
        row = idx // n_cols
        col = idx % n_cols
        
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + title_height + gap)
        
        # title background
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + title_height), (60, 60, 60), -1)
        
        # title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(name, font, 0.7, 2)[0]
        text_x = x + (cell_w - text_size[0]) // 2
        text_y = y + (title_height + text_size[1]) // 2
        cv2.putText(canvas, name, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # image
        canvas[y + title_height : y + title_height + cell_h, x : x + cell_w] = img
        
        # border
        cv2.rectangle(canvas, (x, y + title_height), (x + cell_w, y + title_height + cell_h), (100, 100, 100), 1)
    
    return canvas


def create_demo_horizontal(result, title_height=35, gap=5, max_width=2000):
    # create 1x5 horizontal strip: original → rcc → clahe → denoise → fused
    images = [
        ('original', result.original),
        ('rcc', result.rcc),
        ('clahe', result.clahe),
        ('denoise', result.denoise),
        ('fused', result.fused),
    ]
    
    h, w = result.original.shape[:2]
    n = len(images)
    
    # calculate cell size
    cell_w = (max_width - gap * (n + 1)) // n
    scale = min(1.0, cell_w / w)
    cell_w = int(w * scale)
    cell_h = int(h * scale)
    
    # canvas
    canvas_w = gap + n * (cell_w + gap)
    canvas_h = gap + title_height + cell_h + gap
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # white bg
    
    for idx, (name, img) in enumerate(images):
        x = gap + idx * (cell_w + gap)
        y = gap
        
        # resize
        r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        
        # title
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(name, font, 0.6, 2)[0]
        text_x = x + (cell_w - text_size[0]) // 2
        text_y = y + title_height - 10
        cv2.putText(canvas, name, (text_x, text_y), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # image
        canvas[y + title_height : y + title_height + cell_h, x : x + cell_w] = r
        
        # border
        cv2.rectangle(canvas, (x, y + title_height), (x + cell_w, y + title_height + cell_h), (0, 0, 0), 1)
        
        # arrow between images
        if idx < n - 1:
            arrow_x = x + cell_w + gap // 2
            arrow_y = y + title_height + cell_h // 2
            cv2.arrowedLine(canvas, (arrow_x - 8, arrow_y), (arrow_x + 8, arrow_y), (100, 100, 100), 2, tipLength=0.5)
    
    return canvas


def create_demo_vertical(result, title_width=100, gap=10, max_height=1200):
    # create 5x1 vertical strip
    images = [
        ('original', result.original),
        ('rcc', result.rcc),
        ('clahe', result.clahe),
        ('denoise', result.denoise),
        ('fused', result.fused),
    ]
    
    h, w = result.original.shape[:2]
    n = len(images)
    
    # calculate cell size
    cell_h = (max_height - gap * (n + 1)) // n
    scale = min(1.0, cell_h / h)
    cell_w = int(w * scale)
    cell_h = int(h * scale)
    
    # canvas
    canvas_w = gap + title_width + cell_w + gap
    canvas_h = gap + n * (cell_h + gap)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    for idx, (name, img) in enumerate(images):
        x = gap + title_width
        y = gap + idx * (cell_h + gap)
        
        r = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        
        # title (rotated)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_x = gap + 10
        text_y = y + cell_h // 2 + 5
        cv2.putText(canvas, name, (text_x, text_y), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # image
        canvas[y : y + cell_h, x : x + cell_w] = r
        cv2.rectangle(canvas, (x, y), (x + cell_w, y + cell_h), (0, 0, 0), 1)
    
    return canvas


def process_demo(input_path, output_dir, config=None, layout='grid'):
    # run fusion and create demo visualization
    bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[error] could not read: {input_path}")
        return None
    
    base = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"processing: {input_path}")
    print(f"   size: {bgr.shape[1]}x{bgr.shape[0]}")
    
    # run fusion
    result = run_fusion_pipeline(bgr, config, verbose=True)
    
    # create demo image
    if layout == 'grid':
        demo = create_demo_grid(result)
    elif layout == 'horizontal':
        demo = create_demo_horizontal(result)
    elif layout == 'vertical':
        demo = create_demo_vertical(result)
    else:
        demo = create_demo_grid(result)
    
    # save outputs
    demo_path = os.path.join(output_dir, f"{base}_demo.jpg")
    cv2.imwrite(demo_path, demo, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"   saved demo: {demo_path}")
    
    # also save individual images
    cv2.imwrite(os.path.join(output_dir, f"{base}_0_original.jpg"), result.original)
    cv2.imwrite(os.path.join(output_dir, f"{base}_1_rcc.jpg"), result.rcc)
    cv2.imwrite(os.path.join(output_dir, f"{base}_2_clahe.jpg"), result.clahe)
    cv2.imwrite(os.path.join(output_dir, f"{base}_3_denoise.jpg"), result.denoise)
    cv2.imwrite(os.path.join(output_dir, f"{base}_4_fused.jpg"), result.fused)
    print(f"   saved individual images")
    
    return demo_path


def main():
    parser = argparse.ArgumentParser(description='generate fusion pipeline demo')
    parser.add_argument('input', nargs='+', help='input image(s) or directory')
    parser.add_argument('--output', '-o', default='demo_output', help='output directory')
    parser.add_argument('--layout', '-l', choices=['grid', 'horizontal', 'vertical'], default='grid',
                        help='demo layout (default: grid)')
    parser.add_argument('--max-images', type=int, default=None, help='max images to process')
    
    # fusion params
    parser.add_argument('--rcc-alpha', type=float, default=0.5)
    parser.add_argument('--clahe-clip', type=float, default=6.0)
    parser.add_argument('--clahe-tile', type=int, default=32)
    parser.add_argument('--denoise-gauss', type=int, default=3)
    parser.add_argument('--denoise-median', type=int, default=3)
    parser.add_argument('--denoise-bilateral', type=int, default=3)
    parser.add_argument('--denoise-sharpen', type=float, default=0.1)
    
    # branch scaling
    parser.add_argument('--scale-rcc', type=float, default=1.0, help='rcc branch scale')
    parser.add_argument('--scale-clahe', type=float, default=1.0, help='clahe branch scale')
    parser.add_argument('--scale-denoise', type=float, default=0.2, help='denoise branch scale')
    
    args = parser.parse_args()
    
    config = FusionConfig(
        rcc_alpha=args.rcc_alpha,
        clahe_clip_limit=args.clahe_clip,
        clahe_tile_grid_size=(args.clahe_tile, args.clahe_tile),
        denoise_gaussian_ksize=args.denoise_gauss,
        denoise_median_ksize=args.denoise_median,
        denoise_bilateral_d=args.denoise_bilateral,
        denoise_sharpen_strength=args.denoise_sharpen,
        scale_rcc=args.scale_rcc,
        scale_clahe=args.scale_clahe,
        scale_denoise=args.scale_denoise,
    )
    
    # collect images
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    paths = []
    for s in args.input:
        s = os.path.expanduser(s)
        if os.path.isdir(s):
            for f in sorted(os.listdir(s)):
                if f.lower().endswith(exts):
                    paths.append(os.path.join(s, f))
        elif os.path.isfile(s) and s.lower().endswith(exts):
            paths.append(s)
        else:
            paths.extend(sorted(glob.glob(s)))
    
    if not paths:
        print("[error] no valid images found")
        return
    
    if args.max_images:
        paths = paths[:args.max_images]
    
    print(f"processing {len(paths)} image(s)")
    print(f"layout: {args.layout}")
    print()
    
    for path in paths:
        process_demo(path, args.output, config, args.layout)
        print()
    
    print(f"done. outputs saved to: {args.output}")


if __name__ == '__main__':
    main()