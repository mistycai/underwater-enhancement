# src/contrast/clahe.py

import numpy as np


def _clip_histogram(hist, clip_limit):
    '''
    Clip the histogram and redistribute excess uniformly
    '''
    # clip
    excess = hist - clip_limit
    excess[excess < 0] = 0.0

    hist_clipped = hist.copy()
    hist_clipped[hist_clipped > clip_limit] = clip_limit

    n_excess = excess.sum()
    if n_excess > 0:
        redist = n_excess / hist.size
        hist_clipped += redist

    return hist_clipped


def clahe_luminance(L_norm, tile_size=8, clip_limit=2.0, nbins=256):
    '''
    Apply CLAHE to a single-channel luminance image
    '''
    H, W = L_norm.shape

    #[0, nbins-1] integer bins
    L_scaled = np.clip(L_norm * (nbins - 1), 0, nbins - 1).astype(np.int32)

    L_out = np.zeros_like(L_norm, dtype=np.float32)

    # num tiles in each dimension
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # tile boundaries
            y0 = ty * tile_size
            y1 = min((ty + 1) * tile_size, H)
            x0 = tx * tile_size
            x1 = min((tx + 1) * tile_size, W)

            tile = L_scaled[y0:y1, x0:x1]

            # histogram for this tile
            hist, _ = np.histogram(
                tile.flatten(),
                bins=nbins,
                range=(0, nbins - 1)
            )
            hist = hist.astype(np.float32)

            # clip limit in counts:
            # clip_limit_factor * (tile_area / nbins)
            tile_area = tile.size
            clip_count = clip_limit * (tile_area / nbins)

            hist_clipped = _clip_histogram(hist, clip_count)

            cdf = np.cumsum(hist_clipped)
            cdf = cdf / cdf[-1]  # normalize
            mapped = cdf[tile]
            L_out[y0:y1, x0:x1] = mapped

    L_out = np.clip(L_out, 0.0, 1.0)
    return L_out