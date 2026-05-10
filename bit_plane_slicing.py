# bit_plane_slicing.py
from PIL import Image
import numpy as np

input_path = "bit_plane_slicing.png"
bit = 7   # choose bit plane 0 (LSB) to 7 (MSB)

img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.uint8)
bit_plane = (pixels >> bit) & 1
# Display as 0/255
result_img = bit_plane * 255
result = Image.fromarray(np.uint8(result_img))
result.save(f"bit_plane_{bit}_output.jpg")