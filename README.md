# Medical Image Enhancement Techniques

From-scratch Python implementations of fundamental image enhancement techniques in the spatial domain. All algorithms are implemented using only NumPy and Pillow — no black‑box library functions.

## Techniques Included

### Point Processing (Gray‑level Transformations)
| Technique | File | Description |
|-----------|------|-------------|
| Image Negatives | `negative.py` | Inverts intensity levels, useful for mammography |
| Logarithmic Transformation | `log_transform.py` | Expands dark regions, compresses bright regions |
| Inverse Logarithmic | `inverse_log.py` | Expands bright regions, compresses dark regions |
| Power‑Law (Gamma) | `gamma_transform.py` | Gamma correction for MRI contrast adjustment |

### Contrast Stretching / Linear Transformations
| Technique | File | Description |
|-----------|------|-------------|
| Linear Stretching | `linear_stretch.py` | Full-range min‑max stretching (excludes black background) |
| Piecewise Linear | `piecewise_linear.py` | User‑defined breakpoints for selective enhancement |
| Contrast Stretching | `contrast_stretch.py` | Percentile‑based stretching optimized for chest X‑rays |

### Other Spatial Domain Techniques
| Technique | File | Description |
|-----------|------|-------------|
| Gray‑level Slicing | `gray_level_slicing.py` | Highlights specific intensity ranges |
| Bit‑plane Slicing | `bit_plane_slicing.py` | Extracts individual bit planes for texture analysis |

### Histogram‑based Techniques
| Technique | File | Description |
|-----------|------|-------------|
| Histogram Equalization | `histogram_equalization.py` | Global contrast enhancement via CDF flattening |
| Histogram Matching | `histogram_matching.py` | Matches histogram to a reference image |
| Local Histogram Processing | `local_histogram.py` | Adaptive local equalization for fine details |
| Histogram Statistics | `histogram_stats.py` | Enhancement based on local mean and variance |

## Requirements

```bash
pip install numpy Pillow scipy
```

## Usage
Place your input image in the same folder as the scripts

Update input_path in the script.

Run the script:

```bash
python negative.py
```
Output is saved as <technique>_output.jpg in the same folder

Run All Scripts (Windows CMD)
```cmd
for %f in (*.py) do @python "%f"
```

Provided by navid danaee 8navid@gmail.com
