# histogram_matching.py
from PIL import Image
import numpy as np

source_path = "histogram_matching.png"
reference_path = "histogram_matching_ref.png"
output_path = "histogram_matching_output.png"

# Load both images as grayscale
source_img = Image.open(source_path).convert("L")
reference_img = Image.open(reference_path).convert("L")

source = np.array(source_img, dtype=np.float32)
reference = np.array(reference_img, dtype=np.float32)

height_s, width_s = source.shape
height_r, width_r = reference.shape
total_source = height_s * width_s
total_ref = height_r * width_r

# Compute histogram and CDF for source image
hist_s = np.zeros(256, dtype=np.float32)
for i in range(height_s):
    for j in range(width_s):
        hist_s[int(source[i, j])] += 1

cdf_s = np.zeros(256, dtype=np.float32)
cdf_s[0] = hist_s[0]
for k in range(1, 256):
    cdf_s[k] = cdf_s[k-1] + hist_s[k]

# Compute histogram and CDF for reference image
hist_r = np.zeros(256, dtype=np.float32)
for i in range(height_r):
    for j in range(width_r):
        hist_r[int(reference[i, j])] += 1

cdf_r = np.zeros(256, dtype=np.float32)
cdf_r[0] = hist_r[0]
for k in range(1, 256):
    cdf_r[k] = cdf_r[k-1] + hist_r[k]

# Build mapping: for each source intensity s, find r such that
# CDF_ref(r)/N_ref >= CDF_source(s)/N_source
mapping = np.zeros(256, dtype=np.float32)
r = 0
for s in range(256):
    target = cdf_s[s] / total_source
    while r < 255 and cdf_r[r] / total_ref < target:
        r += 1
    mapping[s] = r

# Apply mapping
matched = np.zeros((height_s, width_s), dtype=np.float32)
for i in range(height_s):
    for j in range(width_s):
        matched[i, j] = mapping[int(source[i, j])]

# Save result
result = Image.fromarray(matched.astype(np.uint8))
result.save(output_path)
print(f"Histogram matched image saved to: {output_path}")