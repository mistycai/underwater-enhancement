import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import logging

from .color_space import bgr_to_lab_luminance, lab_luminance_to_bgr
from .clahe import clahe_luminance, clahe_luminance_fast
from .gating import should_apply_clahe
from .stats import luminance_std, average_gradient

logger = logging.getLogger(__name__)

@dataclass
class CLAHEConfig:
    tile_size: int = 60
    clip_limit: float = 2.0
    nbins: int = 256
    sigma_thresh: float = 0.18
    grad_thresh: float = 0.06
    use_gating: bool = True
    use_fast: bool = True


@dataclass 
class EnhancementResult:
    image: np.ndarray
    applied: bool
    sigma_L: float
    grad_L: float
    sigma_L_out: Optional[float] = None
    grad_L_out: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "applied": self.applied,
            "sigma_L_in": self.sigma_L,
            "grad_L_in": self.grad_L,
            "sigma_L_out": self.sigma_L_out,
            "grad_L_out": self.grad_L_out,
            "contrast_gain": (self.sigma_L_out / self.sigma_L) if self.sigma_L_out and self.sigma_L > 0 else None,
        }


class CLAHEContrastEnhancer:
    def __init__(self, config: Optional[CLAHEConfig] = None, **kwargs):
        if config is None:
            config = CLAHEConfig()
        
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        
        self.config = config
        
    def enhance(self, bgr_img: np.ndarray) -> EnhancementResult:
        # extract luminance
        L_norm, lab_img = bgr_to_lab_luminance(bgr_img)
        
        sigma_L = luminance_std(L_norm)
        grad_L = average_gradient(L_norm)
        
        # gating
        if self.config.use_gating:
            do_enhance = should_apply_clahe(
                L_norm,
                sigma_thresh=self.config.sigma_thresh,
                grad_thresh=self.config.grad_thresh,
            )
        else:
            do_enhance = True
        
        logger.debug(
            f"CLAHE gating: σ_L={sigma_L:.4f}, AG={grad_L:.4f}, "
            f"thresholds=({self.config.sigma_thresh}, {self.config.grad_thresh}), "
            f"apply={do_enhance}"
        )
        
        if not do_enhance:
            return EnhancementResult(
                image=bgr_img.copy(),
                applied=False,
                sigma_L=sigma_L,
                grad_L=grad_L,
            )
        
        # apply CLAHE
        clahe_fn = clahe_luminance_fast if self.config.use_fast else clahe_luminance
        L_enhanced = clahe_fn(
            L_norm,
            tile_size=self.config.tile_size,
            clip_limit=self.config.clip_limit,
            nbins=self.config.nbins,
        )
        
        bgr_out = lab_luminance_to_bgr(L_enhanced, lab_img)
        
        sigma_L_out = luminance_std(L_enhanced)
        grad_L_out = average_gradient(L_enhanced)
        
        logger.debug(
            f"CLAHE applied: sigma_L {sigma_L:.4f} -> {sigma_L_out:.4f} "
            f"({sigma_L_out/sigma_L:.2f}x)"
        )
        
        return EnhancementResult(
            image=bgr_out,
            applied=True,
            sigma_L=sigma_L,
            grad_L=grad_L,
            sigma_L_out=sigma_L_out,
            grad_L_out=grad_L_out,
        )
    
    def __call__(self, bgr_img: np.ndarray) -> np.ndarray:
        return self.enhance(bgr_img).image



#### helpers
def apply_clahe_to_bgr(
    bgr_img: np.ndarray,
    use_gating: bool = True,
    tile_size: int = 60,
    clip_limit: float = 2.0,
    sigma_thresh: float = 0.18,
    grad_thresh: float = 0.06,
) -> Tuple[np.ndarray, Dict]:
    config = CLAHEConfig(
        tile_size=tile_size,
        clip_limit=clip_limit,
        sigma_thresh=sigma_thresh,
        grad_thresh=grad_thresh,
        use_gating=use_gating,
    )
    
    enhancer = CLAHEContrastEnhancer(config)
    result = enhancer.enhance(bgr_img)
    
    return result.image, result.to_dict()


def enhance_for_detection(
    bgr_img: np.ndarray,
    tile_size: int = 60,
    clip_limit: float = 2.0,
) -> np.ndarray:
    config = CLAHEConfig(
        tile_size=tile_size,
        clip_limit=clip_limit,
        use_gating=False,
    )
    enhancer = CLAHEContrastEnhancer(config)
    return enhancer(bgr_img)



##### CLI 

def process_single_image(input_path: str, output_path: str, verbose: bool = True):
    bgr = cv2.imread(input_path)
    if bgr is None:
        print(f"Error: Could not read image: {input_path}")
        return False
    
    enhanced, info = apply_clahe_to_bgr(bgr, use_gating=True)
    
    if verbose:
        print(f"Input:  {input_path}")
        print(f"  sigma_L = {info['sigma_L_in']:.4f}, AG = {info['grad_L_in']:.4f}")
        print(f"  CLAHE applied: {info['applied']}")
        if info['applied']:
            print(f"  sigma_L' = {info['sigma_L_out']:.4f} (gain: {info['contrast_gain']:.2f}x)")
    
    cv2.imwrite(output_path, enhanced)
    
    if verbose:
        print(f"Output: {output_path}")
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Apply CLAHE contrast enhancement to underwater images"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--tile-size", type=int, default=60,
                        help="Tile size in pixels (default: 60)")
    parser.add_argument("--clip-limit", type=float, default=2.0,
                        help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--no-gating", action="store_true",
                        help="Disable detection-aware gating")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    bgr = cv2.imread(args.input)
    if bgr is None:
        print(f"Error: Could not read {args.input}")
        return 1
    
    config = CLAHEConfig(
        tile_size=args.tile_size,
        clip_limit=args.clip_limit,
        use_gating=not args.no_gating,
    )
    
    enhancer = CLAHEContrastEnhancer(config)
    result = enhancer.enhance(bgr)
    
    cv2.imwrite(args.output, result.image)
    
    print(f"{'Applied' if result.applied else 'Skipped'}: {args.input} -> {args.output}")
    if result.applied:
        gain = result.sigma_L_out / result.sigma_L
        print(f"  Contrast gain: {gain:.2f}x")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())