from dataclasses import dataclass
from typing import Tuple

@dataclass
class CLAHEConfig:
    # CLAHE params
    clip_limit: float = 2.0                     # alpha_c
    tile_grid_size: Tuple[int, int] = (8, 8)    # t by t
    n_bins: int = 256

    # color space
    use_lab: bool = True    # LAB-L vs HSV-V

    tau_c: float = 0.17 # sigma_L threshold
    tau_g: float = 0.20 # AG(L) threshold

    enable_gating: bool = True  # False: always apply CLAHE