"""
viz_patches.py – 可视化前 N 个病人的肿瘤中心 patch（轴位/冠位/矢位）+ mask overlay。

用法：
    python viz_patches.py \
        --csv       /path/to/labels.csv \
        --ct_dir    /path/to/ct_npy \
        --mask_dir  /path/to/masks \
        [--label_col recurrence] \
        [--ct_axis_order ZYX] \
        [--n 4] \
        [--out viz_patches.png]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from config import PATCH_SIZE, PATCH_MM, SPACING
from dataset import parse_labels, load_npy, volume_center
from utils import extract_patch, normalise_ct, get_roi_center


MASK_SUFFIX = "_mtv_cervix.npy"   # case-insensitive match


def build_mask_index(mask_dir: str) -> dict:
    """
    Scan mask_dir for files whose name ends with _mtv_cervix.npy
    (case-insensitive). Returns {patient_id_lower: full_path}.
    """
    import glob
    index = {}
    for path in glob.glob(os.path.join(mask_dir, "*.npy")):
        fname = os.path.basename(path)
        if fname.lower().endswith(MASK_SUFFIX):
            pid = fname[: len(fname) - len(MASK_SUFFIX)]   # strip suffix (original case)
            index[pid.lower()] = path
    return index


def find_mask_path(mask_index: dict, pid: str) -> str | None:
    return mask_index.get(pid.lower())


# ── Overlay helpers ────────────────────────────────────────────────────────────

def ct_to_rgb(ct_slice: np.ndarray) -> np.ndarray:
    """[0,1] CT slice → (H,W,3) float32 grayscale RGB."""
    ct = np.clip(ct_slice, 0.0, 1.0).astype(np.float32)
    return np.stack([ct, ct, ct], axis=-1)


def blend_mask(rgb: np.ndarray, mask_slice: np.ndarray,
               color=(1.0, 0.15, 0.15), alpha: float = 0.45) -> np.ndarray:
    """Blend a coloured mask over an RGB image in-place."""
    rgb = rgb.copy()
    m = mask_slice > 0.5
    for c, v in enumerate(color):
        rgb[..., c][m] = alpha * v + (1 - alpha) * rgb[..., c][m]
    return rgb


def show_slice(ax, ct_slice, mask_slice, title="", aspect=1.0):
    """Draw one 2-D slice with mask overlay on *ax*."""
    rgb = blend_mask(ct_to_rgb(ct_slice), mask_slice)
    ax.imshow(rgb, origin="lower", aspect=aspect, interpolation="nearest")
    # Contour outline on top for clarity
    if mask_slice.max() > 0:
        ax.contour(mask_slice, levels=[0.5], colors=["red"], linewidths=[0.8])
    ax.set_title(title, fontsize=7, pad=2)
    ax.axis("off")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            required=True)
    p.add_argument("--ct_dir",         required=True)
    p.add_argument("--mask_dir",       default=None)
    p.add_argument("--label_col",      default="recurrence",
                   choices=["recurrence", "figo"])
    p.add_argument("--ct_axis_order",  default="ZYX", choices=["ZYX", "XYZ"])
    p.add_argument("--mask_axis_order", default="XYZ", choices=["ZYX", "XYZ"],
                   help="Axis order of mask .npy files (default XYZ = no transpose). "
                        "Set to ZYX if masks are stored the same way as CT.")
    p.add_argument("--ct_wl",  type=float, default=40.0)
    p.add_argument("--ct_ww",  type=float, default=400.0)
    p.add_argument("--n",      type=int,   default=4,
                   help="Number of patients to visualise (default: 4).")
    p.add_argument("--out",    default="viz_patches.png")
    args = p.parse_args()

    # ── Patient list ──────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    df = parse_labels(df, args.label_col)
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    # Keep only patients whose CT file exists
    df = df[df["Patient_ID"].apply(
        lambda pid: os.path.exists(os.path.join(args.ct_dir, f"{pid}_CT.npy"))
    )].reset_index(drop=True)

    if len(df) == 0:
        print("[viz] No patients found – check --csv and --ct_dir.")
        return

    mask_index = build_mask_index(args.mask_dir) if args.mask_dir else {}
    print(f"[viz] patch_size={PATCH_SIZE}  "
          f"({PATCH_MM[0]:.0f}×{PATCH_MM[1]:.0f}×{PATCH_MM[2]:.0f} mm)  "
          f"| mask index: {len(mask_index)} entries")

    # Collect the first n patients that have a mask
    selected = []
    for _, record in df.iterrows():
        pid = str(record["Patient_ID"])
        mp  = find_mask_path(mask_index, pid) if args.mask_dir else None
        if mp is not None:
            selected.append((record, mp))
            print(f"  [found {len(selected)}/{args.n}] {pid} → {os.path.basename(mp)}")
        if len(selected) == args.n:
            break

    n = len(selected)
    if n == 0:
        print("[viz] No patients with masks found. Check --mask_dir and file naming.")
        return
    print(f"[viz] Visualising {n} patients with masks.")

    # Physical aspect ratios for non-square slices
    # PATCH_SIZE / SPACING are both in (X, Y, Z) order
    sx, sy, sz = SPACING
    asp_axial    = sy / sx          # X-Y plane  (≈1 if isotropic in-plane)
    asp_coronal  = sz / sx          # X-Z plane  (thick slices → tall)
    asp_sagittal = sz / sy          # Y-Z plane

    # ── Figure layout: n rows × 3 cols ────────────────────────────────────────
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2 + 0.6),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.04})
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Axial (X-Y, mid-Z)", "Coronal (X-Z, mid-Y)", "Sagittal (Y-Z, mid-X)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold", pad=4)

    for row, (record, mask_path) in enumerate(selected):
        pid   = str(record["Patient_ID"])
        label = int(record["label"])

        # Load CT
        ct_vol = load_npy(os.path.join(args.ct_dir, f"{pid}_CT.npy"),
                          args.ct_axis_order)

        # Load mask (guaranteed to exist)
        mask_vol = load_npy(mask_path, args.mask_axis_order)
        mask_vol = (mask_vol > 0).astype(np.float32)
        center   = get_roi_center(mask_vol)
        print(f"  [{pid}] nonzero vox={int(mask_vol.sum())}  centre={center}")

        # Extract patches  →  shape (X, Y, Z) = PATCH_SIZE
        ct_p   = normalise_ct(extract_patch(ct_vol,   center, PATCH_SIZE),
                               wl=args.ct_wl, ww=args.ct_ww)
        mask_p = extract_patch(mask_vol, center, PATCH_SIZE)

        W, H, D = ct_p.shape          # (72, 72, 24)
        cx, cy, cz = W // 2, H // 2, D // 2

        mask_pct  = 100.0 * mask_p.sum() / mask_p.size
        row_label = f"{pid}  |  label={label}  |  mask {mask_pct:.1f}% vox"

        # Axial: ct_p[:, :, cz]  →  transpose so X→columns, Y→rows
        show_slice(axes[row, 0],
                   ct_p[:, :, cz].T,
                   mask_p[:, :, cz].T,
                   title=f"z={cz}/{D}  [{row_label}]",
                   aspect=asp_axial)

        # Coronal: ct_p[:, cy, :]  →  X→col, Z→row
        show_slice(axes[row, 1],
                   ct_p[:, cy, :].T,
                   mask_p[:, cy, :].T,
                   title=f"y={cy}/{H}",
                   aspect=asp_coronal)

        # Sagittal: ct_p[cx, :, :]  →  Y→col, Z→row
        show_slice(axes[row, 2],
                   ct_p[cx, :, :].T,
                   mask_p[cx, :, :].T,
                   title=f"x={cx}/{W}",
                   aspect=asp_sagittal)

    # Legend + suptitle
    red_patch = mpatches.Patch(facecolor=(1, 0.15, 0.15), alpha=0.6,
                                label="Tumour mask (fill + contour)")
    fig.legend(handles=[red_patch], loc="upper right", fontsize=8, framealpha=0.8)
    fig.suptitle(
        f"Tumour-centred patches  |  "
        f"size = {PATCH_SIZE[0]}×{PATCH_SIZE[1]}×{PATCH_SIZE[2]} vox  "
        f"({PATCH_MM[0]:.0f}×{PATCH_MM[1]:.0f}×{PATCH_MM[2]:.0f} mm)",
        fontsize=10, y=1.01,
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {args.out}")


if __name__ == "__main__":
    import sys
    # If no arguments given, use default paths from run.sh
    if len(sys.argv) == 1:
        sys.argv += [
            "--csv",           "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv",
            "--ct_dir",        "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy",
            "--mask_dir",      "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled",
            "--ct_axis_order", "ZYX",
            "--mask_axis_order", "ZYX",
            "--ct_wl",         "40",
            "--ct_ww",         "400",
            "--n",             "4",
            "--out",           "viz_patches.png",
        ]
    main()
