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
    H, W = L_norm.shape
    L_scaled = np.clip(L_norm * nbins, 0, nbins - 1).astype(np.int32)
    
    n_tiles_y = (H + tile_size - 1) // tile_size
    n_tiles_x = (W + tile_size - 1) // tile_size
    
    # store CDFs for all tiles (for interpolation)
    tile_cdfs = np.zeros((n_tiles_y, n_tiles_x, nbins), dtype=np.float32)
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0, y1 = ty * tile_size, min((ty + 1) * tile_size, H)
            x0, x1 = tx * tile_size, min((tx + 1) * tile_size, W)
            tile = L_scaled[y0:y1, x0:x1]
            
            hist, _ = np.histogram(tile.flatten(), bins=nbins, range=(0, nbins))
            hist = hist.astype(np.float32)
            
            tile_area = tile.size
            clip_count = clip_limit * (tile_area / nbins)
            hist_clipped = _clip_histogram(hist, clip_count)
            
            cdf = np.cumsum(hist_clipped)
            cdf = cdf / (cdf[-1] + 1e-8)
            tile_cdfs[ty, tx] = cdf
    
    # bilinear interpolation
    L_out = np.zeros_like(L_norm, dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            # tile center coordinates
            ty_f = (y + 0.5) / tile_size - 0.5
            tx_f = (x + 0.5) / tile_size - 0.5
            
            ty0 = int(np.floor(ty_f))
            tx0 = int(np.floor(tx_f))
            ty1 = ty0 + 1
            tx1 = tx0 + 1
            
            ty0 = max(0, min(ty0, n_tiles_y - 1))
            ty1 = max(0, min(ty1, n_tiles_y - 1))
            tx0 = max(0, min(tx0, n_tiles_x - 1))
            tx1 = max(0, min(tx1, n_tiles_x - 1))
            
            # interpolation weights
            wy = ty_f - np.floor(ty_f)
            wx = tx_f - np.floor(tx_f)
            wy = max(0, min(wy, 1))
            wx = max(0, min(wx, 1))
            
            bin_idx = L_scaled[y, x]
            
            # bilinear blend of 4 neighboring tile CDFs
            v00 = tile_cdfs[ty0, tx0, bin_idx]
            v01 = tile_cdfs[ty0, tx1, bin_idx]
            v10 = tile_cdfs[ty1, tx0, bin_idx]
            v11 = tile_cdfs[ty1, tx1, bin_idx]
            
            v0 = v00 * (1 - wx) + v01 * wx
            v1 = v10 * (1 - wx) + v11 * wx
            L_out[y, x] = v0 * (1 - wy) + v1 * wy
    
    return np.clip(L_out, 0.0, 1.0)