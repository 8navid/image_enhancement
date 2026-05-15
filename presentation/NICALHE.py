# niclahe.py
from PIL import Image
import numpy as np

def niclahe(image, tile_size=8, base_clip_limit=2.0, alpha=1.0):
    """
    Neighbourhood-based Improved Contrast Limited Adaptive Histogram Equalization.

    Parameters:
        image (numpy array): 2D grayscale image (0-255, float or int).
        tile_size (int): size of the square tiles (default 8).
        base_clip_limit (float): baseline clip limit (multiplied by adaptive factor).
        alpha (float): strength of the adaptive adjustment (higher = more sensitive to local contrast).

    Returns:
        numpy array: enhanced image (same shape, dtype float).
    """
    img = image.astype(np.float32)
    h, w = img.shape

    # Pad image so that tiles cover the whole image evenly
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    padded_h, padded_w = img_padded.shape
    tiles_y = padded_h // tile_size
    tiles_x = padded_w // tile_size

    # Compute standard deviation for each tile (and its neighbourhood)
    # First compute tile-wise standard deviation (full padded image)
    tile_stds = np.zeros((tiles_y, tiles_x), dtype=np.float32)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * tile_size
            y1 = y0 + tile_size
            x0 = tx * tile_size
            x1 = x0 + tile_size
            tile = img_padded[y0:y1, x0:x1]
            tile_stds[ty, tx] = np.std(tile)

    # For each tile, compute the mean standard deviation in a 3x3 neighbourhood
    # (mirror pad for border tiles)
    padded_std = np.pad(tile_stds, ((1,1), (1,1)), mode='reflect')
    neighbourhood_std = np.zeros_like(tile_stds)
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            patch = padded_std[ty:ty+3, tx:tx+3]
            neighbourhood_std[ty, tx] = np.mean(patch)

    # Global mean and std of all tile neighbourhood stds for normalisation
    global_mean_std = np.mean(neighbourhood_std)
    global_std_std = np.std(neighbourhood_std) + 1e-6

    # ---- Process each tile ----
    mappings = []   # list of lists, each element: 2D array of shape (256,) mapping
    for ty in range(tiles_y):
        row_mappings = []
        for tx in range(tiles_x):
            y0 = ty * tile_size
            y1 = y0 + tile_size
            x0 = tx * tile_size
            x1 = x0 + tile_size
            tile = img_padded[y0:y1, x0:x1]

            # Adaptive clip limit
            norm_std = (neighbourhood_std[ty, tx] - global_mean_std) / global_std_std
            # Sigmoid to map to a factor around 1.0: low std -> higher clip limit
            factor = 1.0 + alpha * (1.0 - 2.0 / (1.0 + np.exp(-norm_std)))  # range approx 1-alpha to 1+alpha
            clip_limit = base_clip_limit * factor

            # Histogram of this tile
            hist, _ = np.histogram(tile, bins=256, range=(0, 255))
            hist = hist.astype(np.float32)

            # Clip and redistribute
            total_excess = 0
            for k in range(256):
                if hist[k] > clip_limit:
                    total_excess += hist[k] - clip_limit
                    hist[k] = clip_limit

            # Redistribute excess evenly among all bins
            if total_excess > 0:
                add = total_excess / 256.0
                hist += add

            # CDF and equalization mapping
            cdf = np.cumsum(hist)
            cdf_min = cdf[cdf > 0][0] if np.any(cdf > 0) else 0
            total_pixels = tile_size * tile_size
            mapping = np.zeros(256, dtype=np.float32)
            if total_pixels != cdf_min:
                mapping = np.clip((cdf - cdf_min) / (total_pixels - cdf_min) * 255, 0, 255)
            else:
                mapping = np.arange(256, dtype=np.float32)

            row_mappings.append(mapping)
        mappings.append(row_mappings)

    # ---- Bilinear interpolation of mappings ----
    result = np.zeros_like(img_padded)
    for i in range(padded_h):
        for j in range(padded_w):
            # Find four nearest tile centers
            # Tile indices for the tile the pixel belongs to (upper-left tile if fractional)
            ty = i / tile_size - 0.5
            tx = j / tile_size - 0.5

            # Integer indices of the four surrounding tile centers
            ty0 = int(np.floor(ty))
            tx0 = int(np.floor(tx))
            ty1 = ty0 + 1
            tx1 = tx0 + 1

            # Clamp to valid range
            ty0 = max(0, min(ty0, tiles_y - 1))
            ty1 = max(0, min(ty1, tiles_y - 1))
            tx0 = max(0, min(tx0, tiles_x - 1))
            tx1 = max(0, min(tx1, tiles_x - 1))

            # Interpolation weights
            wy = ty - ty0
            wx = tx - tx0
            wy = 1.0 - wy  # weight for upper tiles (distance to lower center)
            wx = 1.0 - wx

            # Pixel intensity
            val = img_padded[i, j]
            val_int = int(val)
            # Fraction for linear interpolation of mapping values
            frac = val - val_int
            # Look-up value from the four mappings (interpolate within mapping if needed)
            def lookup(mapping, val_int, frac):
                if val_int < 255:
                    return mapping[val_int] * (1 - frac) + mapping[val_int + 1] * frac
                else:
                    return mapping[255]

            v00 = lookup(mappings[ty0][tx0], val_int, frac)
            v01 = lookup(mappings[ty0][tx1], val_int, frac)
            v10 = lookup(mappings[ty1][tx0], val_int, frac)
            v11 = lookup(mappings[ty1][tx1], val_int, frac)

            # Bilinear blend
            new_val = (wy * wx * v00 +
                       wy * (1 - wx) * v01 +
                       (1 - wy) * wx * v10 +
                       (1 - wy) * (1 - wx) * v11)
            result[i, j] = new_val

    # Crop back to original size
    result = result[:h, :w]
    return result.astype(np.uint8)

if __name__ == "__main__":
    input_path = "retina.jpg"        # change to your medical image
    output_path = "niclahe_enhanced.jpg"

    img = Image.open(input_path).convert("L")
    img_np = np.array(img)

    enhanced = niclahe(img_np, tile_size=16, base_clip_limit=2.5, alpha=0.8)

    Image.fromarray(enhanced).save(output_path)
    print(f"NICLAHE enhanced image saved to {output_path}")