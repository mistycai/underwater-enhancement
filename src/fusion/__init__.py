"""
Fusion package for underwater image enhancement.

Multi-scale Laplacian pyramid fusion with Ancuti-style weights.
"""

from .multiscale_fusion import (
    multi_scale_fusion_three,
    multi_scale_fusion_two,
    visualize_weights,
    compute_weights_ancuti,
)

from .pyramids import (
    build_gaussian_pyramid,
    build_laplacian_pyramid,
    collapse_laplacian_pyramid,
)

from .weights import (
    compute_laplacian_contrast_weight,
    compute_local_contrast_weight,
    compute_saliency_weight,
    compute_exposedness_weight,
    compute_saturation_weight,
    compute_weights,
    compute_weights_ancuti,
    to_float01,
)

__all__ = [
    # Main fusion functions
    'multi_scale_fusion_three',
    'multi_scale_fusion_two',
    
    # Pyramid functions
    'build_gaussian_pyramid',
    'build_laplacian_pyramid',
    'collapse_laplacian_pyramid',
    
    # Weight functions
    'compute_weights',
    'compute_weights_ancuti',
    'compute_laplacian_contrast_weight',
    'compute_local_contrast_weight',
    'compute_saliency_weight',
    'compute_exposedness_weight',
    'compute_saturation_weight',
    'visualize_weights',
    'to_float01',
]