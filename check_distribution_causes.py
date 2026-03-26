"""
check_distribution_causes.py
Investigate WHY paired (CT+PET) and CT-only patients have different
CT embedding distributions, by examining:

  Step 1 — HU intensity distribution (soft tissue window)
  Step 2 — Volume shape statistics (slice count, FOV)
  Step 3 — CSV metadata comparison (clinical fields)
  Step 4 — Tumour size comparison via mask (skipped if no mask available)

All outputs saved to OUT_DIR.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Hard-coded paths ──────────────────────────────────────────────────────────
CSV      = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR   = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"
LABEL_COL = "recurrence"   # "recurrence" or "figo"

OUT_DIR  = "./outputs/distribution_causes"

# Number of patients to sample per group for HU histogram (use None for all)
HU_SAMPLE_N   = None
# Max voxels sampled per patient for HU histogram
HU_MAX_VOXELS = 20000

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load CSV and split into paired / CT-only ──────────────────────────────────

def load_groups():
    from dataset import parse_labels
    df = pd.read_csv(CSV)
    df = parse_labels(df, label_col=LABEL_COL)
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    df["has_ct"]  = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(CT_DIR,  f"{p}_CT.npy")))
    df["has_pet"] = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy")))
    df = df[df["has_ct"]].copy()
    df_paired = df[df["has_pet"]].copy().reset_index(drop=True)
    df_ctonly = df[~df["has_pet"]].copy().reset_index(drop=True)
    print(f"Paired  : {len(df_paired)} patients")
    print(f"CT-only : {len(df_ctonly)} patients")
    return df_paired, df_ctonly


# ── Build mask index (same logic as dataset.py) ───────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — HU intensity distribution
# ══════════════════════════════════════════════════════════════════════════════

def step1_hu(df_paired, df_ctonly):
    from utils import extract_patch, get_roi_center
    from dataset import load_npy

    # 120mm patch at 0.97×0.97×3.0mm spacing → (124, 124, 40) voxels
    PATCH_SIZE = (124, 124, 40)
    mask_index = build_mask_index(MASK_DIR)

    print("\n" + "=" * 60)
    print("  Step 1: Patch-level HU Distribution (120mm ROI patch)")
    print("=" * 60)

    def collect_patch_hu(df, label, n_sample):
        pids = df["Patient_ID"].tolist()
        if n_sample:
            pids = pids[:n_sample]
        all_vals, means, n_no_mask = [], [], 0
        for pid in pids:
            ct_path = os.path.join(CT_DIR, f"{pid}_CT.npy")
            if not os.path.exists(ct_path):
                continue
            ct = load_npy(ct_path, axis_order='ZYX').astype(np.float32)

            # get ROI center from mask; fallback to volume center
            mask_path = mask_index.get(pid.lower())
            if mask_path:
                mask = load_npy(mask_path, axis_order='XYZ')
                center = get_roi_center(mask)
            else:
                n_no_mask += 1
                center = tuple(s // 2 for s in ct.shape)

            patch = extract_patch(ct, center, PATCH_SIZE)   # (124, 124, 40)

            # raw HU within soft tissue window for histogram
            vals = patch[(patch > -200) & (patch < 200)].flatten()
            if len(vals) > HU_MAX_VOXELS:
                vals = np.random.choice(vals, HU_MAX_VOXELS, replace=False)
            if len(vals) > 0:
                all_vals.append(vals)
                means.append(vals.mean())

        arr = np.concatenate(all_vals)
        print(f"  {label}: patch mean={arr.mean():.2f}  std={arr.std():.2f}  "
              f"median={np.median(arr):.2f}  "
              f"(n={len(all_vals)}, no_mask={n_no_mask} → volume centre used)")
        return arr, means

    np.random.seed(42)
    hu_p, means_p = collect_patch_hu(df_paired, "Paired ", HU_SAMPLE_N)
    hu_c, means_c = collect_patch_hu(df_ctonly, "CT-only", HU_SAMPLE_N)

    ks_stat, ks_p = stats.ks_2samp(
        np.random.choice(hu_p, min(50000, len(hu_p)), replace=False),
        np.random.choice(hu_c, min(50000, len(hu_c)), replace=False),
    )
    t_stat, t_p = stats.ttest_ind(means_p, means_c)
    print(f"\n  KS test (patch HU) : statistic={ks_stat:.4f}  p={ks_p:.4e}")
    print(f"  t-test (patch mean): t={t_stat:.3f}  p={t_p:.4f}")
    if ks_p < 0.05:
        print("  → Significant patch HU difference within ROI.")
    else:
        print("  → No significant patch HU difference → tissue composition similar.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.hist(hu_p, bins=120, alpha=0.5, density=True, color="steelblue",
            label=f"paired (n={len(means_p)})")
    ax.hist(hu_c, bins=120, alpha=0.5, density=True, color="tomato",
            label=f"CT-only (n={len(means_c)})")
    ax.set_xlabel("HU (patch, soft tissue -200~200)")
    ax.set_ylabel("Density")
    ax.set_title(f"Patch HU distribution\nKS={ks_stat:.3f}  p={ks_p:.2e}")
    ax.legend()

    ax = axes[1]
    ax.boxplot([means_p, means_c], tick_labels=["Paired", "CT-only"],
               patch_artist=True, boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_ylabel("Per-patient mean patch HU")
    ax.set_title(f"Per-patient mean patch HU\nt-test p={t_p:.4f}")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "step1_patch_hu_distribution.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Volume shape statistics
# ══════════════════════════════════════════════════════════════════════════════

def step2_shapes(df_paired, df_ctonly):
    print("\n" + "=" * 60)
    print("  Step 2: Volume Shape Statistics")
    print("=" * 60)

    def collect_shapes(df, label):
        shapes = []
        for pid in df["Patient_ID"]:
            path = os.path.join(CT_DIR, f"{pid}_CT.npy")
            if not os.path.exists(path): continue
            shapes.append(np.load(path, mmap_mode='r').shape)
        arr = np.array(shapes)
        print(f"  {label}:")
        print(f"    Z (slices) : mean={arr[:,0].mean():.1f}  std={arr[:,0].std():.1f}  "
              f"min={arr[:,0].min()}  max={arr[:,0].max()}")
        print(f"    Y (rows)   : mean={arr[:,1].mean():.1f}  std={arr[:,1].std():.1f}  "
              f"min={arr[:,1].min()}  max={arr[:,1].max()}")
        print(f"    X (cols)   : mean={arr[:,2].mean():.1f}  std={arr[:,2].std():.1f}  "
              f"min={arr[:,2].min()}  max={arr[:,2].max()}")
        return arr

    shapes_p = collect_shapes(df_paired, "Paired ")
    shapes_c = collect_shapes(df_ctonly, "CT-only")

    # t-test per dimension
    print()
    for i, dim in enumerate(["Z (slices)", "Y (rows)", "X (cols)"]):
        t, p = stats.ttest_ind(shapes_p[:, i], shapes_c[:, i])
        print(f"  t-test {dim}: p={p:.4f}" +
              ("  ← significant" if p < 0.05 else ""))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    dim_names = ["Z — slice count", "Y — rows (FOV)", "X — cols (FOV)"]
    for i, (ax, name) in enumerate(zip(axes, dim_names)):
        ax.boxplot([shapes_p[:, i], shapes_c[:, i]],
                   tick_labels=["Paired", "CT-only"], patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        _, p = stats.ttest_ind(shapes_p[:, i], shapes_c[:, i])
        ax.set_title(f"{name}\nt-test p={p:.4f}")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "step2_shapes.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — CSV metadata comparison
# ══════════════════════════════════════════════════════════════════════════════

def step3_metadata(df_paired, df_ctonly):
    print("\n" + "=" * 60)
    print("  Step 3: CSV Metadata Comparison")
    print("=" * 60)

    df_paired = df_paired.copy(); df_paired["group"] = "paired"
    df_ctonly = df_ctonly.copy(); df_ctonly["group"] = "ctonly"
    combined  = pd.concat([df_paired, df_ctonly], ignore_index=True)

    # Exclude non-informative columns
    skip_cols = {"Patient_ID", "label", "has_ct", "has_pet", "group",
                 "Recurrence", "FIGO 2018 Stage"}
    meta_cols = [c for c in combined.columns if c not in skip_cols]

    if not meta_cols:
        print("  No additional metadata columns found in CSV.")
        return

    print(f"  Metadata columns: {meta_cols}\n")
    rows = []
    for col in meta_cols:
        col_p = df_paired[col].dropna()
        col_c = df_ctonly[col].dropna()
        if col_p.empty or col_c.empty:
            continue

        if pd.api.types.is_numeric_dtype(combined[col]):
            t, p = stats.ttest_ind(col_p, col_c, equal_var=False)
            print(f"  {col:<30s}  paired={col_p.mean():.2f}±{col_p.std():.2f}  "
                  f"ctonly={col_c.mean():.2f}±{col_c.std():.2f}  t-test p={p:.4f}"
                  + ("  ←" if p < 0.05 else ""))
            rows.append({"column": col, "type": "numeric",
                         "paired_mean": col_p.mean(), "ctonly_mean": col_c.mean(),
                         "p_value": p})
        else:
            # categorical: chi-square on value_counts
            all_cats = set(col_p.unique()) | set(col_c.unique())
            counts_p = col_p.value_counts().reindex(all_cats, fill_value=0)
            counts_c = col_c.value_counts().reindex(all_cats, fill_value=0)
            table = np.array([counts_p.values, counts_c.values])
            if table.shape[1] >= 2 and table.min() >= 0:
                try:
                    chi2, p, _, _ = stats.chi2_contingency(table)
                    print(f"  {col:<30s}  [categorical]  chi2={chi2:.2f}  p={p:.4f}"
                          + ("  ←" if p < 0.05 else ""))
                    rows.append({"column": col, "type": "categorical",
                                 "paired_mean": None, "ctonly_mean": None,
                                 "p_value": p})
                except Exception:
                    pass

    if rows:
        out_csv = os.path.join(OUT_DIR, "step3_metadata.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  Saved → {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Tumour size via mask
# ══════════════════════════════════════════════════════════════════════════════

def step4_tumour_size(df_paired, df_ctonly):
    print("\n" + "=" * 60)
    print("  Step 4: Tumour Size (mask voxel count)")
    print("=" * 60)

    mask_index = build_mask_index(MASK_DIR)
    if not mask_index:
        print("  No masks found — skipping.")
        return

    def collect_sizes(df, label):
        sizes = []
        n_missing = 0
        for pid in df["Patient_ID"]:
            path = mask_index.get(pid.lower())
            if path is None:
                n_missing += 1
                continue
            m = np.load(path)
            sizes.append(int((m > 0).sum()))
        print(f"  {label}: {len(sizes)} masks found, {n_missing} missing")
        return sizes

    sizes_p = collect_sizes(df_paired, "Paired ")
    sizes_c = collect_sizes(df_ctonly, "CT-only")

    if not sizes_p or not sizes_c:
        print("  Insufficient mask data for comparison.")
        return

    print(f"\n  Paired  tumour voxels: mean={np.mean(sizes_p):.0f}  "
          f"std={np.std(sizes_p):.0f}  median={np.median(sizes_p):.0f}")
    print(f"  CT-only tumour voxels: mean={np.mean(sizes_c):.0f}  "
          f"std={np.std(sizes_c):.0f}  median={np.median(sizes_c):.0f}")

    t, p = stats.ttest_ind(sizes_p, sizes_c)
    mw_stat, mw_p = stats.mannwhitneyu(sizes_p, sizes_c, alternative='two-sided')
    print(f"\n  t-test         : p={p:.4f}" + ("  ← significant" if p < 0.05 else ""))
    print(f"  Mann-Whitney U : p={mw_p:.4f}" + ("  ← significant" if mw_p < 0.05 else ""))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.boxplot([sizes_p, sizes_c], tick_labels=["Paired", "CT-only"], patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6))
    ax.set_ylabel("Tumour voxel count")
    ax.set_title(f"Tumour size\nt-test p={p:.4f}  MW p={mw_p:.4f}")

    ax = axes[1]
    ax.hist(sizes_p, bins=40, alpha=0.5, density=True,
            color="steelblue", label=f"Paired (n={len(sizes_p)})")
    ax.hist(sizes_c, bins=40, alpha=0.5, density=True,
            color="tomato",    label=f"CT-only (n={len(sizes_c)})")
    ax.set_xlabel("Tumour voxel count")
    ax.set_ylabel("Density")
    ax.set_title("Tumour size distribution")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "step4_tumour_size.png")
    plt.savefig(out, dpi=150)
    print(f"  Saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_paired, df_ctonly = load_groups()
    step1_hu(df_paired, df_ctonly)
    step2_shapes(df_paired, df_ctonly)
    step3_metadata(df_paired, df_ctonly)
    step4_tumour_size(df_paired, df_ctonly)
    print(f"\nAll outputs saved to: {OUT_DIR}")
