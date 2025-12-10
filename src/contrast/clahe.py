# src/contrast/clahe.py


import numpy as np


def _clip_histogram(hist: np.ndarray, clip_limit: float) -> np.ndarray:
    B = hist.size
    # hist_clipped(b) = min{h_m(b), alpha_c}
    hist_clipped = np.minimum(hist, clip_limit)
    # E_m = sum of excess
    excess = np.sum(np.maximum(hist - clip_limit, 0.0))
    if excess > 0:
        hist_clipped = hist_clipped + (excess / B)
    
    return hist_clipped


def clahe_luminance(
    L_norm: np.ndarray,
    tile_size: int = 8,
    clip_limit: float = 2.0,
    nbins: int = 256
) -> np.ndarray:

    H, W = L_norm.shape
    B = nbins
    t = tile_size
    
    # # of tiles in each dimension
    n_tiles_y = max(1, (H + t - 1) // t)
    n_tiles_x = max(1, (W + t - 1) // t)
    
    L_bins = np.clip(L_norm * (B - 1), 0, B - 1).astype(np.int32)
    
    # tile mappings (LUTs)
    # tile_luts[ty, tx, b] = T_m(b) for tile at position (ty, tx)
    tile_luts = np.zeros((n_tiles_y, n_tiles_x, B), dtype=np.float32)
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * t
            y1 = min((ty + 1) * t, H)
            x0 = tx * t
            x1 = min((tx + 1) * t, W)
            
            tile_bins = L_bins[y0:y1, x0:x1]
            n_pixels = tile_bins.size
            
            if n_pixels == 0:
                tile_luts[ty, tx, :] = np.linspace(0, 1, B)
                continue
            
            # histogram h_m(b)
            hist = np.bincount(tile_bins.ravel(), minlength=B)[:B].astype(np.float32)
            # clip limit in counts
            clip_count = clip_limit * (n_pixels / B)
            hist_clipped = _clip_histogram(hist, clip_count)
            cdf = np.cumsum(hist_clipped)
            
            # T(v) = (H(v) - H_min) / (N - H_min) × (L-1)
            cdf_min = cdf[hist_clipped > 0][0] if np.any(hist_clipped > 0) else 0
            denominator = n_pixels - cdf_min
            
            if denominator > 0:
                lut = (cdf - cdf_min) / denominator
            else:
                lut = np.linspace(0, 1, B)
            
            tile_luts[ty, tx, :] = np.clip(lut, 0, 1)
    
    L_enhanced = _apply_with_interpolation(
        L_bins, tile_luts, t, n_tiles_y, n_tiles_x, H, W
    )
    
    return np.clip(L_enhanced, 0.0, 1.0)


def _apply_with_interpolation(
    L_bins: np.ndarray,
    tile_luts: np.ndarray,
    tile_size: int,
    n_tiles_y: int,
    n_tiles_x: int,
    H: int,
    W: int
) -> np.ndarray:
   
    L_out = np.zeros((H, W), dtype=np.float32)
    t = tile_size
    
    for y in range(H):
        for x in range(W):
            # ty has center at (ty + 0.5) × t
            # pixel at y: ty_f = y/t - 0.5
            ty_f = y / t - 0.5
            tx_f = x / t - 0.5
            
            ty0 = int(np.floor(ty_f))
            tx0 = int(np.floor(tx_f))
            ty1 = ty0 + 1
            tx1 = tx0 + 1
            
            # interpolation weights
            wy = ty_f - ty0
            wx = tx_f - tx0
            
            ty0_c = max(0, min(ty0, n_tiles_y - 1))
            ty1_c = max(0, min(ty1, n_tiles_y - 1))
            tx0_c = max(0, min(tx0, n_tiles_x - 1))
            tx1_c = max(0, min(tx1, n_tiles_x - 1))
            
            wy = max(0.0, min(1.0, wy))
            wx = max(0.0, min(1.0, wx))
            
            # get bin index for this pixel
            b = L_bins[y, x]
            
            # look up mapped values from four tiles
            v00 = tile_luts[ty0_c, tx0_c, b]
            v01 = tile_luts[ty0_c, tx1_c, b]
            v10 = tile_luts[ty1_c, tx0_c, b]
            v11 = tile_luts[ty1_c, tx1_c, b]
            
            # bilinear interpolation
            v0 = v00 * (1 - wx) + v01 * wx  # top row
            v1 = v10 * (1 - wx) + v11 * wx  # bottom row
            L_out[y, x] = v0 * (1 - wy) + v1 * wy
    
    return L_out


def clahe_luminance_fast(
    L_norm: np.ndarray,
    tile_size: int = 8,
    clip_limit: float = 2.0,
    nbins: int = 256
) -> np.ndarray:

    H, W = L_norm.shape
    B = nbins
    t = tile_size
    
    n_tiles_y = max(1, (H + t - 1) // t)
    n_tiles_x = max(1, (W + t - 1) // t)
    
    L_bins = np.clip(L_norm * (B - 1), 0, B - 1).astype(np.int32)
    
    # tile LUTs
    tile_luts = np.zeros((n_tiles_y, n_tiles_x, B), dtype=np.float32)
    
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            y0 = ty * t
            y1 = min((ty + 1) * t, H)
            x0 = tx * t
            x1 = min((tx + 1) * t, W)
            
            tile_bins = L_bins[y0:y1, x0:x1]
            n_pixels = tile_bins.size
            
            if n_pixels == 0:
                tile_luts[ty, tx, :] = np.linspace(0, 1, B)
                continue
            
            hist = np.bincount(tile_bins.ravel(), minlength=B)[:B].astype(np.float32)
            clip_count = clip_limit * (n_pixels / B)
            hist_clipped = _clip_histogram(hist, clip_count)
            
            cdf = np.cumsum(hist_clipped)
            
            # normalization
            cdf_min = cdf[hist_clipped > 0][0] if np.any(hist_clipped > 0) else 0
            denominator = n_pixels - cdf_min
            
            if denominator > 0:
                lut = (cdf - cdf_min) / denominator
            else:
                lut = np.linspace(0, 1, B)
            
            tile_luts[ty, tx, :] = np.clip(lut, 0, 1)
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    ty_f = yy / t - 0.5
    tx_f = xx / t - 0.5
    
    ty0 = np.floor(ty_f).astype(np.int32)
    tx0 = np.floor(tx_f).astype(np.int32)
    ty1 = ty0 + 1
    tx1 = tx0 + 1
    
    wy = np.clip(ty_f - ty0, 0.0, 1.0).astype(np.float32)
    wx = np.clip(tx_f - tx0, 0.0, 1.0).astype(np.float32)
    
    ty0 = np.clip(ty0, 0, n_tiles_y - 1)
    ty1 = np.clip(ty1, 0, n_tiles_y - 1)
    tx0 = np.clip(tx0, 0, n_tiles_x - 1)
    tx1 = np.clip(tx1, 0, n_tiles_x - 1)
    
    # interpolation
    L_out = np.zeros((H, W), dtype=np.float32)
    
    for y in range(H):
        for x in range(W):
            b = L_bins[y, x]
            
            v00 = tile_luts[ty0[y, x], tx0[y, x], b]
            v01 = tile_luts[ty0[y, x], tx1[y, x], b]
            v10 = tile_luts[ty1[y, x], tx0[y, x], b]
            v11 = tile_luts[ty1[y, x], tx1[y, x], b]
            
            w_x = wx[y, x]
            w_y = wy[y, x]
            
            v0 = v00 * (1 - w_x) + v01 * w_x
            v1 = v10 * (1 - w_x) + v11 * w_x
            L_out[y, x] = v0 * (1 - w_y) + v1 * w_y
    
    return np.clip(L_out, 0.0, 1.0)