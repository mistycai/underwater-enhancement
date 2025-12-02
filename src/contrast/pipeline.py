# src/contrast/pipeline.py
import cv2
import numpy as np

from .color_space import bgr_to_lab_luminance, lab_luminance_to_bgr
from .clahe import clahe_luminance
from .gating import should_apply_clahe


class CLAHEContrastEnhancer:
    def __init__(
        self,
        tile_size=8,
        clip_limit=2.0,
        nbins=256,
        sigma_thresh=0.10,
        grad_thresh=0.02,
    ):
        self.tile_size = tile_size
        self.clip_limit = clip_limit
        self.nbins = nbins
        self.sigma_thresh = sigma_thresh
        self.grad_thresh = grad_thresh

    def enhance_bgr(self, bgr_img):
        L_norm, lab_img = bgr_to_lab_luminance(bgr_img)
        from .stats import luminance_std, average_gradient
        sigma_L = luminance_std(L_norm)
        grad_L = average_gradient(L_norm)

        # decide whether to apply CLAHE
        do_clahe = should_apply_clahe(
            L_norm,
            sigma_thresh=self.sigma_thresh,
            grad_thresh=self.grad_thresh,
        )

        info = {
            "applied_clahe": bool(do_clahe),
            "sigma_L": float(sigma_L),
            "grad_L": float(grad_L),
            "sigma_thresh": self.sigma_thresh,
            "grad_thresh": self.grad_thresh,
            "reason_sigma": sigma_L < self.sigma_thresh,
            "reason_grad": grad_L < self.grad_thresh,
        }

        print("\n[CLAHE DEBUG INFO]")
        print(f"  sigma_L       = {sigma_L:.4f}")
        print(f"  AG(L)         = {grad_L:.4f}")
        print(f"  alpha_thresh  = {self.sigma_thresh}")
        print(f"  grad_thresh   = {self.grad_thresh}")
        print(f"  sigma < thresh?   = {sigma_L < self.sigma_thresh}")
        print(f"  AG < thresh?  = {grad_L < self.grad_thresh}")
        print(f"  APPLY CLAHE?  = {do_clahe}")
        print("--------------------------------------------------")

        if not do_clahe:
            return bgr_img.copy(), info

        L_enh = clahe_luminance(
            L_norm,
            tile_size=self.tile_size,
            clip_limit=self.clip_limit,
            nbins=self.nbins,
        )

        # back to BGR
        bgr_out = lab_luminance_to_bgr(L_enh, lab_img)
        return bgr_out, info


def apply_clahe_to_bgr(bgr_img, use_gating=True):
    """
    helper function to apply CLAHE:
    from src.contrast.pipeline import apply_clahe_to_bgr
    """
    if use_gating:
        enhancer = CLAHEContrastEnhancer(
            tile_size=8,
            clip_limit=2.0,
            nbins=256,
            sigma_thresh=0.20,
            grad_thresh=0.25,
        )
    else:
        # if gating disabled, set very large thresholds so it always applies
        enhancer = CLAHEContrastEnhancer(
            tile_size=8,
            clip_limit=2.0,
            nbins=256,
            sigma_thresh=999.0,
            grad_thresh=999.0,
        )

    return enhancer.enhance_bgr(bgr_img)


def demo_on_image(input_path, output_path):
    bgr = cv2.imread(input_path)
    if bgr is None:
        print("Could not read image:", input_path)
        return

    out, info = apply_clahe_to_bgr(bgr, use_gating=True)

    print("CLAHE applied:", info["applied_clahe"])
    cv2.imwrite(output_path, out)
    print("Saved to:", output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 -m src.contrast.pipeline input.jpg output.jpg")
    else:
        demo_on_image(sys.argv[1], sys.argv[2])