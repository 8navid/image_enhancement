# negative.py
from PIL import Image
import numpy as np

input_path = "negative.png"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)
L = 256
neg_pixels = (L - 1) - pixels
result = Image.fromarray(np.uint8(neg_pixels))
result.save("negative_output.jpg")