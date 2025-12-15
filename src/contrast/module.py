import cv2
import numpy as np
from .config import CLAHEConfig
from .clahe import apply_clahe_luminance
from .gating import compute_luminance_stats, should_apply_clahe, soft_gate_mask

class ContrastEnhancerCLAHE:
    def __init__(self, config: CLAHEConfig | None = None, use_soft_gate: bool = False):
        self.config = config or CLAHEConfig()
        self.use_soft_gate = use_soft_gate

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        return self.enhance(img_bgr)

    def enhance(self, img_bgr: np.ndarray) -> np.ndarray:
        sigma_L, AG = compute_luminance_stats(img_bgr, self.config)

        if not self.config.enable_gating:
            # always apply CLAHE
            return apply_clahe_luminance(img_bgr, self.config)

        # hard gating rule
        if not should_apply_clahe(sigma_L, AG, self.config):
            return img_bgr

        if not self.use_soft_gate:
            return apply_clahe_luminance(img_bgr, self.config)

        # soft spatial gating
        gate = soft_gate_mask(img_bgr, self.config)
        gate_3c = cv2.merge([gate, gate, gate])   # H×W×3

        clahe_img = apply_clahe_luminance(img_bgr, self.config)
        blended = gate_3c * clahe_img.astype(np.float32) + \
                  (1 - gate_3c) * img_bgr.astype(np.float32)

        return blended.astype(np.uint8)