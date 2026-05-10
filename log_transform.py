# log_transform.py
from PIL import Image
import numpy as np

input_path = "log_transform.png"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

c = 255 / np.log(1 + np.max(pixels))   # scale to fill [0,255]
log_pixels = c * np.log(1 + pixels)
result = Image.fromarray(np.uint8(log_pixels))
result.save("log_output.jpg")