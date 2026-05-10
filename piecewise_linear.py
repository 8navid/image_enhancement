# piecewise_linear.py
from PIL import Image
import numpy as np

input_path = "piecewise_linear.jpg"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

# Define breakpoints (r1,s1) and (r2,s2) – here as example:
r1, s1 = 70, 0       # map dark values to 0
r2, s2 = 180, 255    # map bright values to 255
# For r < r1: s = (s1/r1) * r
# For r1 <= r <= r2: s = s1 + ((r - r1) * (s2 - s1) / (r2 - r1))
# For r > r2: s = s2 + ((r - r2) * (255 - s2) / (255 - r2))

def piecewise(r):
    if r < r1:
        return (s1 / r1) * r
    elif r <= r2:
        return s1 + (r - r1) * (s2 - s1) / (r2 - r1)
    else:
        return s2 + (r - r2) * (255 - s2) / (255 - r2)

vec_piecewise = np.vectorize(piecewise)
result_pixels = vec_piecewise(pixels)
result = Image.fromarray(np.uint8(result_pixels))
result.save("piecewise_linear_output.jpg")