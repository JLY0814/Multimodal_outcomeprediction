"""
visualize_npy.py
Load two .npy CT volumes and display mid-slices along all three axes
to verify orientation and pixel integrity.

Set FILE_A and FILE_B below, then run:
  python visualize_npy.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Files to compare ──────────────────────────────────────────────────────────
FILE_A = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/GYNDATASET001_CT.npy"
FILE_B = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/GYNDATASET065_CT.npy"

OUT = "./ct_visualize.png"

# CT display window (HU)
WL, WW = 40, 400   # window level / width
VMIN = WL - WW // 2
VMAX = WL + WW // 2

# ─────────────────────────────────────────────────────────────────────────────

def mid_slices(vol):
    """Return axial / coronal / sagittal mid-slices (each 2-D)."""
    d0, d1, d2 = vol.shape
    axial    = vol[d0 // 2, :, :]   # mid along axis-0
    coronal  = vol[:, d1 // 2, :]   # mid along axis-1
    sagittal = vol[:, :, d2 // 2]   # mid along axis-2
    return axial, coronal, sagittal

def load_and_report(path, label):
    arr = np.load(path)
    print(f"[{label}]  path  : {path}")
    print(f"         shape : {arr.shape}  dtype={arr.dtype}")
    print(f"         mean  : {arr.mean():.1f}   std={arr.std():.1f}"
          f"   min={arr.min():.1f}   max={arr.max():.1f}")
    return arr

print("=" * 60)
vol_a = load_and_report(FILE_A, "A")
print()
vol_b = load_and_report(FILE_B, "B")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
titles = ["Axis-0 mid (axial?)", "Axis-1 mid (coronal?)", "Axis-2 mid (sagittal?)"]

for row, (vol, label) in enumerate([(vol_a, "A"), (vol_b, "B")]):
    slices = mid_slices(vol)
    for col, (sl, title) in enumerate(zip(slices, titles)):
        ax = axes[row][col]
        ax.imshow(sl, cmap="gray", vmin=VMIN, vmax=VMAX, origin="upper")
        ax.set_title(f"{label}: {title}\nslice shape={sl.shape}", fontsize=8)
        ax.axis("off")

# Label rows with file names
for row, path in enumerate([FILE_A, FILE_B]):
    name = path.split("/")[-1]
    axes[row][0].set_ylabel(name, fontsize=7, rotation=0,
                            labelpad=5, ha="right", va="center")

plt.suptitle("CT volume mid-slice comparison\n"
             f"A shape={vol_a.shape}    B shape={vol_b.shape}", fontsize=10)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"\nFigure saved → {OUT}")
