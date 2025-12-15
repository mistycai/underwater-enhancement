import cv2
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, 'src')

if len(sys.argv) < 4:
    print("Usage: python fusion_enhance_images.py INPUT_DIR OUTPUT_DIR CONFIG_TYPE")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
config_type = sys.argv[3]


RCC_ALPHA = float(os.getenv('RCC_ALPHA', '0.5'))
CLAHE_CLIP = float(os.getenv('CLAHE_CLIP', '6.0'))
CLAHE_TILE = int(os.getenv('CLAHE_TILE', '32'))
DENOISE_GAUSS = int(os.getenv('DENOISE_GAUSS', '3'))
DENOISE_MEDIAN = int(os.getenv('DENOISE_MEDIAN', '3'))
DENOISE_BILATERAL = int(os.getenv('DENOISE_BILATERAL', '3'))
DENOISE_SHARPEN = float(os.getenv('DENOISE_SHARPEN', '0.1'))

print(f"Input dir: {input_dir}")
print(f"Output dir: {output_dir}")
print(f"Config: {config_type}")
print(f"Parameters: RCC α={RCC_ALPHA}, CLAHE clip={CLAHE_CLIP} tile={CLAHE_TILE}, "
      f"Denoise g={DENOISE_GAUSS} m={DENOISE_MEDIAN} b={DENOISE_BILATERAL} s={DENOISE_SHARPEN}")


def apply_rcc(bgr, alpha=0.5):
    try:
        from color_correction.rcc_wrcc import rcc_rgb
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Use basic RCC, not WRCC
        rgb_out = rcc_rgb(rgb, alpha=alpha)
        return cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    except ImportError as e:
        print(f"Warning: Cannot import RCC: {e}")
        print("Using fallback implementation")
        # Fallback: simple red channel boost
        result = bgr.astype(np.float32)
        b, g, r = cv2.split(result)
        
        mean_g = np.mean(g)
        mean_r = np.mean(r)
        
        if mean_r > 0:
            scale = np.power(mean_g / (mean_r + 1e-6), alpha)
            r = np.clip(r * scale, 0, 255)
        
        result = cv2.merge([b, g, r])
        return result.astype(np.uint8)


def apply_clahe(bgr, clip=6.0, tile=32):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def apply_denoise(bgr, gauss=3, median=3, bilateral=3, sharpen=0.1):
    result = bgr.copy()
    
    # Gaussian blur
    if gauss > 0:
        ksize = gauss if gauss % 2 == 1 else gauss + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 1.5)
    
    # Median filter
    if median > 0:
        ksize = median if median % 2 == 1 else median + 1
        result = cv2.medianBlur(result, ksize)
    
    # Bilateral filter
    if bilateral > 0:
        result = cv2.bilateralFilter(result, bilateral, 25.0, 25.0)
    
    # Laplacian sharpening
    if sharpen > 0:
        f = result.astype(np.float32)
        lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
        result = np.clip(f - sharpen * lap, 0, 255).astype(np.uint8)
    
    return result


def build_gaussian_pyramid(img, levels):
    pyramid = [img.astype(np.float32)]
    for _ in range(levels):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid


def build_laplacian_pyramid(img, levels):
    gaussian = build_gaussian_pyramid(img, levels)
    laplacian = []
    
    for i in range(levels):
        size = (gaussian[i].shape[1], gaussian[i].shape[0])
        upsampled = cv2.pyrUp(gaussian[i+1], dstsize=size)
        lap = gaussian[i] - upsampled
        laplacian.append(lap)
    
    laplacian.append(gaussian[-1])
    return laplacian


def collapse_pyramid(pyramid):
    result = pyramid[-1]
    
    for i in range(len(pyramid) - 2, -1, -1):
        size = (pyramid[i].shape[1], pyramid[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = result + pyramid[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def multi_scale_fusion(images, levels=5):
    if len(images) == 0:
        return None
    if len(images) == 1:
        return images[0]
    
    # Build Laplacian pyramids for each input
    pyramids = [build_laplacian_pyramid(img, levels) for img in images]
    
    # Fuse at each level (simple average)
    fused_pyramid = []
    for level in range(levels + 1):
        level_images = [pyramid[level] for pyramid in pyramids]
        fused_level = np.mean(level_images, axis=0)
        fused_pyramid.append(fused_level)
    
    return collapse_pyramid(fused_pyramid)


images = sorted(Path(input_dir).glob("*.jpg"))
print(f"Found {len(images)} images")

if len(images) == 0:
    print("ERROR: No images found!")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)


for img_path in tqdm(images, desc=f"Processing {config_type}"):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Warning: Cannot read {img_path}")
        continue
    
    if config_type == "none":
        result = bgr
    
    elif config_type == "rcc":
        result = apply_rcc(bgr, RCC_ALPHA)
    
    elif config_type == "clahe":
        result = apply_clahe(bgr, CLAHE_CLIP, CLAHE_TILE)
    
    elif config_type == "denoise":
        result = apply_denoise(bgr, DENOISE_GAUSS, DENOISE_MEDIAN, 
                              DENOISE_BILATERAL, DENOISE_SHARPEN)
    
    elif config_type == "2input":
 
        input1 = apply_rcc(bgr, RCC_ALPHA)
        input2 = apply_clahe(bgr, CLAHE_CLIP, CLAHE_TILE)
        result = multi_scale_fusion([input1, input2], levels=5)
    
    elif config_type == "3input":

        input1 = apply_rcc(bgr, RCC_ALPHA)
        input2 = apply_clahe(bgr, CLAHE_CLIP, CLAHE_TILE)
        input3 = apply_denoise(bgr, DENOISE_GAUSS, DENOISE_MEDIAN, 
                              DENOISE_BILATERAL, DENOISE_SHARPEN)
        result = multi_scale_fusion([input1, input2, input3], levels=5)
    
    else:
        print(f"Warning: Unknown config type '{config_type}', using raw")
        result = bgr
    
    out_path = os.path.join(output_dir, img_path.name)
    cv2.imwrite(out_path, result)

print(f"✓ Saved {len(images)} images to {output_dir}")