"""
check_mask_axis.py
Scan all mask .npy files and report shape statistics to detect axis order issues.

Expected mask shape: (X, Y, Z) where Z (slice axis) is the smallest dimension.
Suspicious: shape[0] > shape[2]  →  Z may be at the end (like wrong CT files).
"""

import os
import glob
import numpy as np

MASK_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/masks_MTV_PTV_resampled_0.9766_3.0mm"

# ─────────────────────────────────────────────────────────────────────────────

mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.npy")))
print(f"Found {len(mask_files)} mask files in:\n  {MASK_DIR}\n")

n_ok    = 0
n_wrong = 0
n_square = 0  # shape[0] == shape[2], ambiguous

shapes_ok    = []
shapes_wrong = []

for path in mask_files:
    fname = os.path.basename(path)
    arr   = np.load(path, mmap_mode='r')
    shape = arr.shape

    if len(shape) != 3:
        print(f"  [SKIP ndim={len(shape)}]  {fname}  shape={shape}")
        continue

    nonzero = np.count_nonzero(arr)
    flag = ""
    if shape[0] < shape[2]:
        n_ok += 1
        shapes_ok.append(shape)
        flag = ""
    elif shape[0] > shape[2]:
        n_wrong += 1
        shapes_wrong.append(shape)
        flag = "  ← SUSPICIOUS (Z at end?)"
    else:
        n_square += 1
        flag = "  ← ambiguous (square)"

    print(f"  {fname:<50s}  shape={str(shape):<20s}  nonzero={nonzero}{flag}")

print()
print("=" * 60)
print(f"  Total scanned     : {len(mask_files)}")
print(f"  Likely correct    : {n_ok}   (shape[0] < shape[2])")
print(f"  Suspicious        : {n_wrong}  (shape[0] > shape[2])")
print(f"  Ambiguous/square  : {n_square}")

if shapes_ok:
    print(f"\n  Correct shape examples   : {shapes_ok[:3]}")
if shapes_wrong:
    print(f"  Suspicious shape examples: {shapes_wrong[:3]}")
