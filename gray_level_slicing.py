# gray_level_slicing.py
from PIL import Image
import numpy as np

input_path = "gray_level_slicing.png"
img = Image.open(input_path).convert("L")
pixels = np.array(img, dtype=np.float32)

low, high = 100, 150          # range to highlight
keep_background = False       # True = preserve original outside range

if keep_background:
    sliced = np.where((pixels >= low) & (pixels <= high), 255, pixels)
else:
    sliced = np.where((pixels >= low) & (pixels <= high), 255, 0)

result = Image.fromarray(np.uint8(sliced))
result.save("gray_slicing_output.jpg")