# linear_stretch.py
from PIL import Image
import numpy as np

input_path = "linear_stretch.jpg"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

# Exclude black background pixels (value = 0) from min/max calculation
foreground = pixels[pixels > 0]
if len(foreground) == 0:
    foreground = pixels  # fallback if entire image is black

min_val, max_val = np.min(foreground), np.max(foreground)
if max_val > min_val:
    stretched = (pixels - min_val) * (255.0 / (max_val - min_val))
else:
    stretched = pixels.copy()
result = Image.fromarray(np.uint8(stretched))
result.save("linear_stretch_output.jpg")