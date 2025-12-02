# src/fusion/run_fusion_demo.py

import os
import cv2

from src.enhance_wrappers import (
    apply_rcc_bgr,
    apply_clahe_bgr,
    apply_denoise_pipeline,
)
from src.fusion.multiscale_fusion import multi_scale_fusion_three


def run_fusion_on_image(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(input_path)
    if bgr is None:
        print("Could not read:", input_path)
        return

    base = os.path.splitext(os.path.basename(input_path))[0]
    H, W = bgr.shape[:2]

    ######## RCC/WRCC  ########
    bgr_rcc = apply_rcc_bgr(bgr, use_wrcc=True)
    bgr_rcc = cv2.resize(bgr_rcc, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, base + "_rcc.jpg"), bgr_rcc)

    ######## CLAHE ########
    bgr_clahe, info = apply_clahe_bgr(bgr)
    print("[CLAHE] applied =", info.get("applied_clahe", True))
    bgr_clahe = cv2.resize(bgr_clahe, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, base + "_clahe.jpg"), bgr_clahe)

    ######## Denoising ########
    bgr_denoise = apply_denoise_pipeline(bgr)
    bgr_denoise = cv2.resize(bgr_denoise, (W, H), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(out_dir, base + "_denoise.jpg"), bgr_denoise)

    ######## Multi-scale fusion (Ancuti weights + pyramids) ########
    fused = multi_scale_fusion_three(
        bgr_rcc,
        bgr_clahe,
        bgr_denoise,
        levels=4,
        lam_contrast=1.0, # can tune these later
        lam_saturation=1.0,
        lam_exposed=1.0,
    )

    fused_path = os.path.join(out_dir, base + "_fused.jpg")
    cv2.imwrite(fused_path, fused)
    print("Saved fused result to:", fused_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 -m src.fusion.run_fusion_demo <input_image> <out_dir>")
        print("Example:")
        print("  python3 -m src.fusion.run_fusion_demo "
              "data/raw/cuttlefish.jpg results/fusion_demo")
    else:
        run_fusion_on_image(sys.argv[1], sys.argv[2])