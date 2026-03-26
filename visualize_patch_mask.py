"""
visualize_patch_mask.py
Show CT patch (120mm ROI) with mask overlay for N paired and N CT-only patients.
Each patient: axial / coronal / sagittal mid-slices of the patch.
Mask contour drawn in red; CT shown in soft-tissue window.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils   import extract_patch, get_roi_center
from dataset import load_npy, parse_labels
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
CSV      = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR   = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"
LABEL_COL = "recurrence"

OUT_DIR  = "./outputs/patch_mask_viz"

# ── Options ───────────────────────────────────────────────────────────────────
N_PAIRED  = 6   # how many paired patients to show
N_CTONLY  = 6   # how many CT-only patients to show
SEED      = 42  # random seed for patient selection

# CT display window
WL, WW = 40, 400
VMIN = WL - WW // 2
VMAX = WL + WW // 2

# 120mm patch at 0.97×0.97×3.0mm → (124, 124, 40)
PATCH_SIZE = (124, 124, 40)

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def build_mask_index(mask_dir):
    index = {}
    if not mask_dir or not os.path.isdir(mask_dir):
        return index
    for path in glob.glob(os.path.join(mask_dir, "*.npy")):
        fname = os.path.basename(path).lower()
        if "_mtv_cervix.npy" in fname:
            pid = fname[: fname.index("_mtv_cervix.npy")]
            index[pid] = path
    return index


def load_patch_and_mask(pid, mask_index):
    """
    Load CT patch (124,124,40) centred on mask centroid.
    Returns (ct_patch, mask_patch, has_mask).
    Falls back to volume centre if no mask.
    """
    ct_path = os.path.join(CT_DIR, f"{pid}_CT.npy")
    ct = load_npy(ct_path, axis_order='ZYX').astype(np.float32)

    mask_path = mask_index.get(pid.lower())
    has_mask  = mask_path is not None
    if has_mask:
        mask   = load_npy(mask_path, axis_order='XYZ').astype(np.float32)
        center = get_roi_center(mask)
        mask_patch = extract_patch(mask, center, PATCH_SIZE)
    else:
        center     = tuple(s // 2 for s in ct.shape)
        mask_patch = np.zeros(PATCH_SIZE, dtype=np.float32)

    ct_patch = extract_patch(ct, center, PATCH_SIZE)
    return ct_patch, mask_patch, has_mask


def mid_slices(ct_patch, mask_patch):
    """Return (axial, coronal, sagittal) tuples of (ct_slice, mask_slice)."""
    d0, d1, d2 = ct_patch.shape   # (124, 124, 40)
    views = [
        (ct_patch[d0 // 2, :, :],  mask_patch[d0 // 2, :, :],  "axial (mid-X)"),
        (ct_patch[:, d1 // 2, :],  mask_patch[:, d1 // 2, :],  "coronal (mid-Y)"),
        (ct_patch[:, :, d2 // 2],  mask_patch[:, :, d2 // 2],  "sagittal (mid-Z)"),
    ]
    return views


def plot_group(pids, mask_index, group_label, out_path):
    n_cols = 3   # axial / coronal / sagittal
    n_rows = len(pids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.2, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, pid in enumerate(pids):
        try:
            ct_patch, mask_patch, has_mask = load_patch_and_mask(pid, mask_index)
        except Exception as e:
            print(f"  [SKIP] {pid}: {e}")
            for col in range(n_cols):
                axes[row, col].axis("off")
            continue

        views = mid_slices(ct_patch, mask_patch)
        mask_sum = int((mask_patch > 0).sum())

        for col, (ct_sl, msk_sl, view_name) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(ct_sl.T, cmap="gray", vmin=VMIN, vmax=VMAX,
                      origin="lower", aspect="auto")
            if has_mask and mask_sum > 0:
                # overlay mask as semi-transparent red fill
                masked = np.ma.masked_where(msk_sl.T == 0, msk_sl.T)
                ax.imshow(masked, cmap=ListedColormap(["red"]),
                          alpha=0.35, origin="lower", aspect="auto")
                # contour outline
                ax.contour(msk_sl.T, levels=[0.5], colors="red",
                           linewidths=0.8, origin="lower")
            ax.set_title(
                f"{pid}\n{view_name}"
                + (f"  mask={mask_sum}vox" if has_mask else "  [no mask]"),
                fontsize=6.5
            )
            ax.axis("off")

    fig.suptitle(f"{group_label}  (patch {PATCH_SIZE}, WL/WW {WL}/{WW})",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(CSV)
    df = parse_labels(df, label_col=LABEL_COL)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["has_ct"]  = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(CT_DIR,  f"{p}_CT.npy")))
    df["has_pet"] = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy")))
    df = df[df["has_ct"]].copy()

    mask_index = build_mask_index(MASK_DIR)

    df_paired = df[df["has_pet"]].copy().reset_index(drop=True)
    df_ctonly = df[~df["has_pet"]].copy().reset_index(drop=True)

    rng = np.random.default_rng(SEED)

    # Prefer patients that have masks; fall back to any patient
    def pick_pids(df, n):
        has_m = df["Patient_ID"].apply(lambda p: p.lower() in mask_index)
        with_mask    = df[has_m]["Patient_ID"].tolist()
        without_mask = df[~has_m]["Patient_ID"].tolist()
        rng.shuffle(with_mask)
        rng.shuffle(without_mask)
        chosen = (with_mask + without_mask)[:n]
        return chosen

    paired_pids = pick_pids(df_paired, N_PAIRED)
    ctonly_pids = pick_pids(df_ctonly, N_CTONLY)

    print(f"Paired  patients selected : {paired_pids}")
    print(f"CT-only patients selected : {ctonly_pids}")

    plot_group(paired_pids, mask_index, f"Paired CT+PET  (n={N_PAIRED})",
               os.path.join(OUT_DIR, "paired_patch_mask.png"))

    plot_group(ctonly_pids, mask_index, f"CT-only  (n={N_CTONLY})",
               os.path.join(OUT_DIR, "ctonly_patch_mask.png"))

    print("Done.")
