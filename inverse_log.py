# inverse_log.py
from PIL import Image
import numpy as np

input_path = "inverse_log.jpg"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

# s = c * (exp(r) - 1)   scaled to [0,255]
max_val = np.max(pixels)
c = 255 / (np.exp(max_val / 255) - 1)   # r normalized to [0,1] first
exp_pixels = c * (np.exp(pixels / 255) - 1)
result = Image.fromarray(np.uint8(exp_pixels))
result.save("inverse_log_output.jpg")