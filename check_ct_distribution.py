"""
check_ct_distribution.py
Verify whether CT-only patients and paired (CT+PET) patients have different
distributions, which would explain why adding CT-only data does not help
improve CT-only AUC in the missing-gate model.

ANALYSIS_MODE:
  1 → Label (prevalence) distribution + Fisher's exact test
  2 → CT embedding distribution (KS test + PCA/t-SNE visualisation)
      Requires a trained checkpoint (CHECKPOINT below).

Usage:
  python check_ct_distribution.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Hard-coded paths (mirror run.sh) ─────────────────────────────────────────
CSV      = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR   = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"
LABEL_COL = "recurrence"           # "recurrence" or "figo"

# Mode 2 only: path to a trained checkpoint
CHECKPOINT = "/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/dual_branch_3d_cnn/spring_output1/aux0.3_drop0_pairvalmissing_gate_mil_recurrence_20260318_011445/fold1_best.pth"

# Output directory for figures / CSVs
OUT_DIR = "./spring_output1/distribution_check"

# ── Select analysis ───────────────────────────────────────────────────────────
ANALYSIS_MODE = 2     # 1 = label distribution,  2 = CT embedding distribution

# ── Mode 2 options ─────────────────────────────────────────────────────────────
USE_TSNE  = False     # True → t-SNE (slow),  False → PCA (fast)
MODEL_SIZE = "base"   # "base" or "small"
USE_MIL   = True     # set True if the checkpoint was trained with --use_mil
DEVICE    = "cuda:0"  # device for embedding extraction

# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


# ── Shared: load + parse CSV, split into paired / CT-only ────────────────────

def load_groups():
    """Return (df_paired, df_ctonly) after label parsing."""
    from dataset import parse_labels
    df = pd.read_csv(CSV)
    df = parse_labels(df, label_col=LABEL_COL)
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    df["has_ct"]  = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(CT_DIR,  f"{p}_CT.npy")))
    df["has_pet"] = df["Patient_ID"].apply(
        lambda p: os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy")))

    df = df[df["has_ct"]].copy()          # keep patients that have at least CT
    df_paired = df[df["has_pet"]].copy()
    df_ctonly = df[~df["has_pet"]].copy()
    return df_paired, df_ctonly


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1 – label distribution
# ══════════════════════════════════════════════════════════════════════════════

def run_mode1():
    df_paired, df_ctonly = load_groups()

    lbl_p = df_paired["label"]
    lbl_c = df_ctonly["label"]

    print("=" * 60)
    print("  Label Distribution Analysis")
    print("=" * 60)
    print(f"  Paired  (CT+PET) : n={len(lbl_p):4d}  "
          f"pos={lbl_p.sum():3d}  neg={len(lbl_p)-lbl_p.sum():3d}  "
          f"prevalence={lbl_p.mean():.3f}")
    print(f"  CT-only          : n={len(lbl_c):4d}  "
          f"pos={lbl_c.sum():3d}  neg={len(lbl_c)-lbl_c.sum():3d}  "
          f"prevalence={lbl_c.mean():.3f}")

    # Fisher's exact test
    table = np.array([
        [int(lbl_p.sum()),  int((lbl_p == 0).sum())],
        [int(lbl_c.sum()),  int((lbl_c == 0).sum())],
    ])
    odds, p_fisher = stats.fisher_exact(table)
    print(f"\n  Fisher's exact test  : OR={odds:.3f},  p={p_fisher:.4f}")
    if p_fisher < 0.05:
        print("  → Significant difference in label distribution (p < 0.05).")
        print("    The two groups are clinically distinct populations.")
    else:
        print("  → No significant difference in label distribution (p ≥ 0.05).")
        print("    Label shift alone does not explain the distribution gap.")

    # Chi-squared test (for completeness)
    chi2, p_chi2, _, _ = stats.chi2_contingency(table)
    print(f"  Chi-squared test     : χ²={chi2:.3f}, p={p_chi2:.4f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    groups = ["Paired\n(CT+PET)", "CT-only"]
    prev   = [lbl_p.mean(), lbl_c.mean()]
    counts = [len(lbl_p), len(lbl_c)]
    bars   = ax.bar(groups, prev, color=["steelblue", "tomato"], alpha=0.8)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"{LABEL_COL.capitalize()} prevalence")
    ax.set_title(f"Label distribution by group\n"
                 f"Fisher p={p_fisher:.4f},  OR={odds:.2f}")
    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "label_distribution.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n  Figure saved → {out_fig}")

    # Save summary CSV
    summary = pd.DataFrame({
        "group":      ["paired", "ct_only"],
        "n":          [len(lbl_p), len(lbl_c)],
        "n_pos":      [int(lbl_p.sum()), int(lbl_c.sum())],
        "n_neg":      [int((lbl_p == 0).sum()), int((lbl_c == 0).sum())],
        "prevalence": [lbl_p.mean(), lbl_c.mean()],
        "fisher_OR":  [odds, odds],
        "fisher_p":   [p_fisher, p_fisher],
    })
    out_csv = os.path.join(OUT_DIR, "label_distribution.csv")
    summary.to_csv(out_csv, index=False)
    print(f"  Summary saved → {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2 – CT embedding distribution
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(model, patient_ids, labels, device):
    """
    Run CT volumes through model.ct_branch + adaptive_avg_pool3d
    to get (N, C) embeddings without touching PET/dose.
    """
    from dataset import PatchDataset
    from config import DLA_CHANNELS, DLA_CHANNELS_SMALL, FUSION_HIDDEN, FUSION_HIDDEN_SMALL

    ds = PatchDataset(
        csv_path       = CSV,
        ct_dir         = CT_DIR,
        pet_dir        = PET_DIR,
        mask_dir       = MASK_DIR,
        label_col      = LABEL_COL,
        patient_ids    = patient_ids,
        ct_axis_order  = "ZYX",
        pet_axis_order = "XYZ",
        ct_wl=40, ct_ww=400, pet_max=200,
        augment        = False,
        use_missing_gate = True,   # allows CT-only patients through
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=4,
                                         shuffle=False, num_workers=2)

    raw_model = model.module if hasattr(model, "module") else model

    all_emb, all_lbl = [], []
    with torch.no_grad():
        for batch in loader:
            # batch: (ct, pet, dose, label, pet_present)  from missing_gate mode
            ct  = batch[0].to(device)          # (B, 1, D, H, W)
            lbl = batch[3].numpy()
            feat = raw_model.ct_branch(ct)                       # (B, C, d, h, w)
            emb  = F.adaptive_avg_pool3d(feat, 1).flatten(1)    # (B, C)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(lbl)

    return np.concatenate(all_emb, axis=0), np.concatenate(all_lbl, axis=0)


def run_mode2():
    from evaluate import load_model
    from config import (DLA_CHANNELS, DLA_CHANNELS_SMALL,
                        FUSION_HIDDEN, FUSION_HIDDEN_SMALL,
                        DOSE_CHANNELS, DOSE_CHANNELS_SMALL)

    if not os.path.exists(CHECKPOINT):
        sys.exit(f"[Error] Checkpoint not found: {CHECKPOINT}\n"
                 "Set CHECKPOINT at the top of this script.")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    _dla_ch = DLA_CHANNELS_SMALL if MODEL_SIZE == "small" else DLA_CHANNELS
    _fus_h  = FUSION_HIDDEN_SMALL if MODEL_SIZE == "small" else FUSION_HIDDEN
    _dos_ch = DOSE_CHANNELS_SMALL if MODEL_SIZE == "small" else DOSE_CHANNELS

    model = load_model(
        CHECKPOINT, device,
        dla_channels=_dla_ch, fusion_hidden=_fus_h, dose_channels=_dos_ch,
        use_mil=USE_MIL,
        use_missing_gate=True,
    )
    model.eval()

    df_paired, df_ctonly = load_groups()
    ids_p = df_paired["Patient_ID"].tolist()
    ids_c = df_ctonly["Patient_ID"].tolist()

    print("Extracting CT embeddings for paired patients …")
    emb_p, lbl_p = extract_embeddings(model, ids_p, df_paired["label"].tolist(), device)
    print(f"  → {emb_p.shape[0]} patients, embedding dim={emb_p.shape[1]}")

    print("Extracting CT embeddings for CT-only patients …")
    emb_c, lbl_c = extract_embeddings(model, ids_c, df_ctonly["label"].tolist(), device)
    print(f"  → {emb_c.shape[0]} patients, embedding dim={emb_c.shape[1]}")

    # ── KS test per embedding dimension ──────────────────────────────────────
    ks_stats = np.array([
        stats.ks_2samp(emb_p[:, i], emb_c[:, i]).statistic
        for i in range(emb_p.shape[1])
    ])
    ks_pvals = np.array([
        stats.ks_2samp(emb_p[:, i], emb_c[:, i]).pvalue
        for i in range(emb_p.shape[1])
    ])
    n_sig = (ks_pvals < 0.05).sum()

    print("\n" + "=" * 60)
    print("  CT Embedding Distribution (KS test per dimension)")
    print("=" * 60)
    print(f"  Embedding dim     : {emb_p.shape[1]}")
    print(f"  Mean KS statistic : {ks_stats.mean():.4f}")
    print(f"  Max  KS statistic : {ks_stats.max():.4f}")
    print(f"  Dims with p<0.05  : {n_sig} / {emb_p.shape[1]} "
          f"({100*n_sig/emb_p.shape[1]:.1f}%)")
    if n_sig > emb_p.shape[1] * 0.1:
        print("  → Substantial embedding distribution gap between groups.")
    else:
        print("  → Embedding distributions are largely similar.")

    # ── Dimensionality reduction ─────────────────────────────────────────────
    all_emb = np.concatenate([emb_p, emb_c], axis=0)
    n_p = len(emb_p)

    if USE_TSNE:
        from sklearn.manifold import TSNE
        print("\nRunning t-SNE (may take a minute) …")
        proj = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(all_emb) - 1)).fit_transform(all_emb)
        method = "t-SNE"
    else:
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(all_emb)
        var  = pca.explained_variance_ratio_
        method = f"PCA (var={var[0]:.2f},{var[1]:.2f})"
        print(f"\nPCA explained variance: PC1={var[0]:.3f}, PC2={var[1]:.3f}")

    proj_p = proj[:n_p]
    proj_c = proj[n_p:]

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: group colouring
    ax = axes[0]
    ax.scatter(proj_p[:, 0], proj_p[:, 1], c="steelblue", alpha=0.6,
               s=30, label=f"Paired (n={len(emb_p)})")
    ax.scatter(proj_c[:, 0], proj_c[:, 1], c="tomato", alpha=0.6,
               s=30, marker="^", label=f"CT-only (n={len(emb_c)})")
    ax.set_title(f"CT embedding — group ({method})")
    ax.legend(fontsize=8)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    # Right: label × group colouring
    ax = axes[1]
    colors = {(0, "p"): "royalblue", (1, "p"): "tomato",
              (0, "c"): "deepskyblue", (1, "c"): "orange"}
    markers = {"p": "o", "c": "^"}
    for lbl_val, grp, proj_g, lbl_g in [
        (0, "p", proj_p, lbl_p), (1, "p", proj_p, lbl_p),
        (0, "c", proj_c, lbl_c), (1, "c", proj_c, lbl_c),
    ]:
        mask = lbl_g == lbl_val
        label_str = ("paired" if grp == "p" else "ct-only") + \
                    (" pos" if lbl_val == 1 else " neg")
        ax.scatter(proj_g[mask, 0], proj_g[mask, 1],
                   c=colors[(lbl_val, grp)], alpha=0.6, s=30,
                   marker=markers[grp], label=label_str)
    ax.set_title(f"CT embedding — label × group ({method})")
    ax.legend(fontsize=7)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")

    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "ct_embedding_distribution.png")
    plt.savefig(out_fig, dpi=150)
    print(f"\n  Figure saved → {out_fig}")

    # KS statistic distribution plot
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(ks_stats, bins=30, color="steelblue", alpha=0.8)
    ax2.axvline(ks_stats.mean(), color="red", linestyle="--",
                label=f"mean={ks_stats.mean():.3f}")
    ax2.set_xlabel("KS statistic (per embedding dim)")
    ax2.set_ylabel("Count")
    ax2.set_title("CT embedding KS test: paired vs CT-only")
    ax2.legend()
    plt.tight_layout()
    out_ks = os.path.join(OUT_DIR, "ks_statistics.png")
    plt.savefig(out_ks, dpi=150)
    print(f"  KS figure saved  → {out_ks}")

    # Save KS results
    ks_df = pd.DataFrame({"dim": np.arange(len(ks_stats)),
                           "ks_stat": ks_stats, "p_value": ks_pvals})
    out_csv = os.path.join(OUT_DIR, "ks_results.csv")
    ks_df.to_csv(out_csv, index=False)
    print(f"  KS CSV saved     → {out_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# Mode 0 – raw volume shape & intensity sanity check
# ══════════════════════════════════════════════════════════════════════════════

def run_mode0(n_samples: int = 10):
    """
    Load the first n_samples CT volumes from each group (paired / CT-only)
    and print their raw shape + intensity statistics.
    If axis order is wrong for one group, shapes or statistics will differ
    systematically (e.g. transposed dims, very different mean/std).
    """
    df_paired, df_ctonly = load_groups()
    ids_p = df_paired["Patient_ID"].tolist()
    ids_c = df_ctonly["Patient_ID"].tolist()

    def sample_stats(pids, label):
        print(f"\n=== {label} (first {min(n_samples, len(pids))}) ===")
        shapes, means, stds, mins, maxs = [], [], [], [], []
        for pid in pids[:n_samples]:
            path = os.path.join(CT_DIR, f"{pid}_CT.npy")
            if not os.path.exists(path):
                print(f"  [{pid}] NOT FOUND")
                continue
            vol = np.load(path)
            shapes.append(vol.shape)
            means.append(vol.mean())
            stds.append(vol.std())
            mins.append(vol.min())
            maxs.append(vol.max())
            print(f"  {pid:<20s}  shape={str(vol.shape):<20s}"
                  f"  mean={vol.mean():7.1f}  std={vol.std():6.1f}"
                  f"  min={vol.min():7.1f}  max={vol.max():7.1f}")

        if shapes:
            unique_shapes = set(shapes)
            print(f"  → Unique shapes : {unique_shapes}")
            print(f"  → Mean of means : {np.mean(means):.1f}   "
                  f"Mean of stds : {np.mean(stds):.1f}")
            print(f"  → Mean min      : {np.mean(mins):.1f}   "
                  f"Mean max     : {np.mean(maxs):.1f}")

    sample_stats(ids_p, f"Paired CT+PET (n_total={len(ids_p)})")
    sample_stats(ids_c, f"CT-only       (n_total={len(ids_c)})")

    print("\nDiagnosis hints:")
    print("  - Same shapes + similar mean/std → axis order is likely consistent.")
    print("  - Transposed dims (e.g. (40,124,124) vs (124,124,40)) → axis mismatch.")
    print("  - Same shapes but very different mean/std → acquisition/normalisation diff.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if ANALYSIS_MODE == 0:
        run_mode0()
    elif ANALYSIS_MODE == 1:
        run_mode1()
    elif ANALYSIS_MODE == 2:
        run_mode2()
    else:
        sys.exit(f"Unknown ANALYSIS_MODE={ANALYSIS_MODE}. Set to 0, 1, or 2.")
