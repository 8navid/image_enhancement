# contrast_stretch.py
from PIL import Image
import numpy as np

input_path = "contrast_stretch.jpg"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

# Exclude black background pixels (value = 0) from percentile calculation
foreground = pixels[pixels > 0]
if len(foreground) == 0:
    foreground = pixels  # fallback if entire image is black

# Stretch only between percentiles (e.g. 5% and 95%) to avoid outliers
low_percentile = np.percentile(foreground, 5)
high_percentile = np.percentile(foreground, 95)
stretched = np.clip((pixels - low_percentile) * (255.0 / (high_percentile - low_percentile)), 0, 255)
result = Image.fromarray(np.uint8(stretched))
result.save("contrast_stretch_output.jpg")