# gamma_transform.py
from PIL import Image
import numpy as np

input_path = "gamma_transform.png"
gamma = 0.5       # change as needed (<1 darkens, >1 brightens)
c = 1.0

img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32) / 255.0
gamma_pixels = c * (pixels ** gamma) * 255
result = Image.fromarray(np.uint8(gamma_pixels))
result.save("gamma_output.jpg")