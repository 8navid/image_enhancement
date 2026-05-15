# histogram_equalization.py
from PIL import Image
import numpy as np

input_path = "histogram_equalization.png"
output_path = "histogram_equalization_output.jpg"

# Load image as grayscale
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

height, width = pixels.shape
total_pixels = height * width

# 1. Compute histogram (256 bins)
hist = np.zeros(256, dtype=np.float32)
for i in range(height):
    for j in range(width):
        val = int(pixels[i, j])
        hist[val] += 1

# 2. Compute cumulative distribution function (CDF)
cdf = np.zeros(256, dtype=np.float32)
cdf[0] = hist[0]
for k in range(1, 256):
    cdf[k] = cdf[k-1] + hist[k]

# 3. Find CDF minimum (first non-zero)
cdf_min = 0.0
for k in range(256):
    if cdf[k] > 0:
        cdf_min = cdf[k]
        break

# 4. Create mapping: T(k) = round((cdf[k] - cdf_min) / (N - cdf_min) * 255)
mapping = np.zeros(256, dtype=np.float32)
if total_pixels != cdf_min:
    for k in range(256):
        mapping[k] = round((cdf[k] - cdf_min) / (total_pixels - cdf_min) * 255)
        if mapping[k] < 0:
            mapping[k] = 0
        if mapping[k] > 255:
            mapping[k] = 255
else:
    for k in range(256):
        mapping[k] = k  # uniform image, keep as is

# 5. Apply mapping
equalized = np.zeros((height, width), dtype=np.float32)
for i in range(height):
    for j in range(width):
        original_val = int(pixels[i, j])
        equalized[i, j] = mapping[original_val]

# Save result
result = Image.fromarray(equalized.astype(np.uint8))
result.save(output_path)
print(f"Histogram equalized image saved to: {output_path}")