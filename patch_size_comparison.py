#!/usr/bin/env python3
"""
patch_size_comparison.py
Compare two sets of MIL models trained with different patch sizes.

Analyses
--------
  [1] AUC comparison         — per-fold bar chart + mean±std + paired t-test
  [2] Prediction correlation — scatter of prob_A vs prob_B (matched patients)
  [3] Attention std          — violin plot comparing per-patient attn_std
  [4] Top-1 removal          — prob drop comparison (Mann-Whitney)
  [5] Center vs periphery    — ablation inside/outside patch core (Wilcoxon)
  [6] Visual comparison      — side-by-side CT/PET mid-slice of top-N cases

Usage
-----
  # Single-fold comparison (fold 1)
  python patch_size_comparison.py single \
      --dir_a outputs/small/ --dir_b outputs/large/ \
      --patch_mm_a 80 80 80 --patch_mm_b 120 120 120 \
      --val_only [data args…]

  # Multi-fold comparison (all 5 folds + aggregate)
  python patch_size_comparison.py multi \
      --dir_a outputs/small/ --dir_b outputs/large/ \
      --patch_mm_a 80 80 80 --patch_mm_b 120 120 120 \
      --val_only [data args…]
"""

import os, sys, glob, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SPACING,
    DLA_CHANNELS, DLA_CHANNELS_SMALL,
    FUSION_HIDDEN, FUSION_HIDDEN_SMALL,
    DOSE_CHANNELS, DOSE_CHANNELS_SMALL,
    MIL_OFFSETS,
)
from evaluate import load_model
from dataset import PatchDataset, parse_labels
from utils import set_seed
from mil_analysis import register_attn_hook, _batched_infer   # reuse existing helpers

GROUPS = ["All", "Recurrence", "No-recurrence"]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _mm_to_vox(patch_mm):
    """Convert physical size (x,y,z mm) to voxel counts using global SPACING."""
    return tuple(int(round(m / s)) for m, s in zip(patch_mm, SPACING))


def _collect_checkpoints(folder):
    paths = sorted(glob.glob(os.path.join(folder, "fold*_best.pth")))
    if not paths:
        raise FileNotFoundError(f"No fold*_best.pth found in: {folder}")
    return paths


def _get_val_ids(csv_path, label_col, fold_idx, num_folds=5, seed=42):
    df  = pd.read_csv(csv_path)
    df  = parse_labels(df, label_col)
    ids = df["Patient_ID"].astype(str).values
    lbs = df["label"].astype(int).values
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for i, (_, val_idx) in enumerate(skf.split(ids, lbs), start=1):
        if i == fold_idx:
            return ids[val_idx].tolist()
    raise ValueError(f"fold_idx={fold_idx} out of range for num_folds={num_folds}")


def _make_dataset(args, patch_mm, mil_offsets, patient_ids=None):
    patch_size = _mm_to_vox(patch_mm)
    return PatchDataset(
        csv_path        = args.csv,
        ct_dir          = args.ct_dir,
        pet_dir         = args.pet_dir or args.ct_dir,
        mask_dir        = args.mask_dir,
        label_col       = args.label_col,
        ct_axis_order   = args.ct_axis_order,
        pet_axis_order  = args.pet_axis_order,
        ct_wl           = args.ct_wl,
        ct_ww           = args.ct_ww,
        pet_max         = args.pet_max,
        use_mil         = True,
        mil_patch_size  = patch_size,
        mil_offsets     = mil_offsets,
        augment         = False,
        patient_ids     = patient_ids,
    )


def _load_model(ckpt, device, args, _dla, _fus, _dose):
    return load_model(
        ckpt, device,
        use_dose      = args.use_dose,
        use_mil       = True,
        dla_channels  = _dla,
        fusion_hidden = _fus,
        dose_channels = _dose,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Attention export (simplified — no CSV, uses dataset.df for pids)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _export_attention(model, dataset, device, batch_size=4,
                      num_workers=4, use_amp=False):
    """
    Run the model on every patient and return records:
      pid, label, prob, attn (N,), ct (N,1,D,H,W), pet, dose
    pid is recovered from dataset.df using a sequential index.
    """
    model.eval()
    cap, handle = register_attn_hook(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"),
                        timeout=120 if num_workers > 0 else 0)
    amp_ctx = (torch.autocast("cuda", dtype=torch.float16)
               if (use_amp and device.type == "cuda") else torch.no_grad())

    records     = []
    patient_idx = 0

    for batch in tqdm(loader, desc="  export attn", leave=False):
        ct, pet, dose, labels = batch[0], batch[1], batch[2], batch[3]
        ct   = ct.to(device)
        pet  = pet.to(device)
        dose = dose.to(device)

        with amp_ctx:
            logits = model(ct, pet, dose)
        probs = torch.sigmoid(logits).float().cpu()   # (B,)
        attns = cap.A[:, :, 0].cpu()                  # (B, N)

        for b in range(ct.size(0)):
            pid = str(dataset.df.iloc[patient_idx]["Patient_ID"])
            records.append({
                "pid":   pid,
                "label": int(labels[b].item()),
                "prob":  float(probs[b].item()),
                "attn":  attns[b].numpy(),          # (N,)
                "ct":    ct[b].cpu(),               # (N, 1, D, H, W)
                "pet":   pet[b].cpu(),
                "dose":  dose[b].cpu(),
            })
            patient_idx += 1

    handle.remove()
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1 – AUC comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_auc_comparison(auc_a_list, auc_b_list, label_a, label_b, out_dir):
    n  = len(auc_a_list)
    xs = np.arange(1, n + 1)
    w  = 0.35

    mean_a, std_a = np.mean(auc_a_list), np.std(auc_a_list)
    mean_b, std_b = np.mean(auc_b_list), np.std(auc_b_list)

    fig, ax = plt.subplots(figsize=(max(6, 2 * n + 2), 5))
    ax.bar(xs - w / 2, auc_a_list, width=w, color="steelblue", alpha=0.85,
           label=label_a)
    ax.bar(xs + w / 2, auc_b_list, width=w, color="tomato",    alpha=0.85,
           label=label_b)
    ax.axhline(mean_a, color="steelblue", ls="--", lw=1.5,
               label=f"{label_a} mean={mean_a:.3f}±{std_a:.3f}")
    ax.axhline(mean_b, color="tomato",    ls="--", lw=1.5,
               label=f"{label_b} mean={mean_b:.3f}±{std_b:.3f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Fold {i}" for i in xs])
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1)
    ax.set_title(f"AUC comparison: {label_a} vs {label_b}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "auc_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

    # Paired t-test (meaningful only when n > 1)
    if n > 1:
        _, p = scipy_stats.ttest_rel(auc_b_list, auc_a_list)
        p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
        print(f"  [1] AUC  {label_a}={mean_a:.3f}±{std_a:.3f}  "
              f"{label_b}={mean_b:.3f}±{std_b:.3f}  paired-t p={p_str}")
    else:
        print(f"  [1] AUC  {label_a}={auc_a_list[0]:.4f}  "
              f"{label_b}={auc_b_list[0]:.4f}")
    print(f"       → {path}")

    pd.DataFrame({
        "fold":              list(range(1, n + 1)),
        f"auc_{label_a}":   auc_a_list,
        f"auc_{label_b}":   auc_b_list,
    }).to_csv(os.path.join(out_dir, "auc_comparison.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2 – Prediction correlation
# ══════════════════════════════════════════════════════════════════════════════

def compare_predictions(recs_a, recs_b, label_a, label_b, out_dir, suffix=""):
    da = {r["pid"]: r for r in recs_a}
    db = {r["pid"]: r for r in recs_b}
    pids = sorted(set(da) & set(db))
    if not pids:
        print("  [2] No matched patients — skipping prediction correlation.")
        return

    p_a  = np.array([da[p]["prob"]  for p in pids])
    p_b  = np.array([db[p]["prob"]  for p in pids])
    lbs  = np.array([da[p]["label"] for p in pids])

    r_val, p_val = scipy_stats.pearsonr(p_a, p_b)
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"

    colors = ["tomato" if l == 1 else "steelblue" for l in lbs]
    m, b_coef = np.polyfit(p_a, p_b, 1)
    xs = np.linspace(p_a.min(), p_a.max(), 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(p_a, p_b, c=colors, alpha=0.6, s=30)
    ax.plot(xs, m * xs + b_coef, "k--", lw=1.2, label="regression")
    lo = min(p_a.min(), p_b.min()) - 0.05
    hi = max(p_a.max(), p_b.max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8, ls=":")
    ax.set_xlabel(f"Prob  ({label_a})")
    ax.set_ylabel(f"Prob  ({label_b})")
    ax.set_title(f"Prediction correlation  (n={len(pids)})\n"
                 f"Pearson r={r_val:.3f},  p={p_str}"
                 + (f"\n{suffix}" if suffix else ""))
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=8, label="Recurrence"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=8, label="No-recurrence"),
        Line2D([0], [0], color="k", ls="--", label="regression"),
    ], fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "prediction_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [2] Prediction correlation r={r_val:.3f} p={p_str} → {path}")

    pd.DataFrame({
        "pid": pids, "label": lbs,
        f"prob_{label_a}": p_a,
        f"prob_{label_b}": p_b,
    }).to_csv(os.path.join(out_dir, "prediction_correlation.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3 – Attention std distribution
# ══════════════════════════════════════════════════════════════════════════════

def compare_attn_std(recs_a, recs_b, label_a, label_b, out_dir, suffix=""):
    std_a = np.array([float(r["attn"].std()) for r in recs_a])
    std_b = np.array([float(r["attn"].std()) for r in recs_b])
    _, p  = scipy_stats.mannwhitneyu(std_b, std_a, alternative="two-sided")
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot([std_a, std_b], positions=[1, 2], showmedians=True)
    colors = ["steelblue", "tomato"]
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    # Overlay individual points
    for xpos, vals in enumerate([std_a, std_b], start=1):
        ax.scatter(np.random.uniform(xpos - 0.08, xpos + 0.08, len(vals)),
                   vals, color="black", alpha=0.3, s=12, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{label_a}\n(n={len(std_a)})",
                        f"{label_b}\n(n={len(std_b)})"])
    ax.set_ylabel("Per-patient attention std")
    ax.set_title(f"Attention std: {label_a}={std_a.mean():.4f}  "
                 f"{label_b}={std_b.mean():.4f}\n"
                 f"Mann-Whitney p={p_str} (two-sided)"
                 + (f"\n{suffix}" if suffix else ""))
    plt.tight_layout()
    path = os.path.join(out_dir, "attn_std_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [3] Attn std  {label_a}={std_a.mean():.4f}  "
          f"{label_b}={std_b.mean():.4f}  MW p={p_str} → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4 – Top-1 removal sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def compare_top1_removal(model_a, recs_a, model_b, recs_b, device,
                          label_a, label_b, out_dir,
                          infer_batch_size=64, use_amp=False, suffix=""):
    def _drops(model, recs):
        model.eval()
        drops = []
        for r in tqdm(recs, desc="    top-1 drop", leave=False):
            top_idx = int(np.argmax(r["attn"]))
            ct_m    = r["ct"].clone();   ct_m[top_idx]   = 0.
            pet_m   = r["pet"].clone();  pet_m[top_idx]  = 0.
            dose_m  = r["dose"].clone(); dose_m[top_idx] = 0.
            with torch.no_grad():
                p_mod = _batched_infer(model, device,
                                       [ct_m], [pet_m], [dose_m],
                                       infer_batch_size, use_amp)
            drops.append(float(r["prob"]) - p_mod[0])
        return drops

    da = _drops(model_a, recs_a)
    db = _drops(model_b, recs_b)
    _, p = scipy_stats.mannwhitneyu(db, da, alternative="two-sided")
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [da, db],
        tick_labels=[f"{label_a}\n(n={len(da)})", f"{label_b}\n(n={len(db)})"],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="black", lw=2),
    )
    bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("tomato");    bp["boxes"][1].set_alpha(0.7)
    for xpos, vals in enumerate([da, db], start=1):
        ax.scatter([xpos + np.random.uniform(-0.08, 0.08) for _ in vals],
                   vals, color="black", alpha=0.3, s=12, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Prob drop  (base − top-1 patch removed)")
    ax.set_title(f"Top-1 removal sensitivity\n"
                 f"{label_a}={np.mean(da):.4f}  {label_b}={np.mean(db):.4f}  "
                 f"MW p={p_str}"
                 + (f"\n{suffix}" if suffix else ""))
    plt.tight_layout()
    path = os.path.join(out_dir, "top1_removal_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [4] Top-1 removal  {label_a}={np.mean(da):.4f}  "
          f"{label_b}={np.mean(db):.4f}  MW p={p_str} → {path}")
    return da, db


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5 – Center vs periphery ablation
# ══════════════════════════════════════════════════════════════════════════════

def _center_mask(D, H, W, inner_frac):
    """Boolean mask (D,H,W): True inside the inner_frac×inner_frac×inner_frac core."""
    dz = max(1, int(D * inner_frac / 2))
    dy = max(1, int(H * inner_frac / 2))
    dx = max(1, int(W * inner_frac / 2))
    m = torch.zeros(D, H, W, dtype=torch.bool)
    m[D//2-dz:D//2+dz, H//2-dy:H//2+dy, W//2-dx:W//2+dx] = True
    return m


def center_periphery_ablation(model, records, device, label, out_dir,
                               inner_frac=0.5, infer_batch_size=64,
                               use_amp=False, suffix=""):
    """
    For each patient's highest-attention patch, compute:
      center_drop   = base_prob - prob(center zeroed)      ← tumor core ablated
      periph_drop   = base_prob - prob(periphery zeroed)   ← peritumoral ablated

    If periph_drop > center_drop the model leverages peritumoral information.
    Wilcoxon signed-rank test: H1 = periph_drop > center_drop.
    """
    model.eval()
    center_drops, periph_drops = [], []

    for r in tqdm(records, desc=f"    center/periph [{label}]", leave=False):
        top_idx       = int(np.argmax(r["attn"]))
        ct0, p0, d0   = r["ct"], r["pet"], r["dose"]
        D, H, W       = ct0.shape[2:]
        cmask         = _center_mask(D, H, W, inner_frac)   # (D,H,W)

        # Center zeroed (core of the top-attention patch)
        ct_c  = ct0.clone();  ct_c[top_idx,  0][cmask]  = 0.
        pet_c = p0.clone();   pet_c[top_idx, 0][cmask]  = 0.
        dos_c = d0.clone();   dos_c[top_idx, 0][cmask]  = 0.

        # Periphery zeroed (outer region of the top-attention patch)
        ct_p  = ct0.clone();  ct_p[top_idx,  0][~cmask] = 0.
        pet_p = p0.clone();   pet_p[top_idx, 0][~cmask] = 0.
        dos_p = d0.clone();   dos_p[top_idx, 0][~cmask] = 0.

        with torch.no_grad():
            probs = _batched_infer(model, device,
                                   [ct_c, ct_p], [pet_c, pet_p], [dos_c, dos_p],
                                   infer_batch_size, use_amp)
        base = float(r["prob"])
        center_drops.append(base - probs[0])
        periph_drops.append(base - probs[1])

    _, p_wil = scipy_stats.wilcoxon(periph_drops, center_drops,
                                     alternative="greater")
    p_str = f"{p_wil:.3f}" if p_wil >= 0.001 else "<0.001"

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [center_drops, periph_drops],
        tick_labels=[f"Center zeroed\n(tumor core, inner {int(inner_frac*100)}%)",
                     f"Periphery zeroed\n(peritumoral, outer {int((1-inner_frac)*100)}%)"],
        patch_artist=True, widths=0.5,
        medianprops=dict(color="black", lw=2),
    )
    bp["boxes"][0].set_facecolor("gold");          bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor("mediumpurple");  bp["boxes"][1].set_alpha(0.8)
    for xpos, vals in enumerate([center_drops, periph_drops], start=1):
        ax.scatter([xpos + np.random.uniform(-0.08, 0.08) for _ in vals],
                   vals, color="black", alpha=0.3, s=12, zorder=3)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.set_ylabel("Prob drop  (base − ablated)")
    ax.set_title(f"Center vs Periphery Ablation  [{label}]\n"
                 f"center={np.mean(center_drops):.4f}  "
                 f"periph={np.mean(periph_drops):.4f}  "
                 f"Wilcoxon(periph>center) p={p_str}"
                 + (f"\n{suffix}" if suffix else ""))
    plt.tight_layout()
    safe = label.replace(" ", "_").replace("/", "")
    path = os.path.join(out_dir, f"center_periphery_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [5] Center/periph [{label}]  center={np.mean(center_drops):.4f}  "
          f"periph={np.mean(periph_drops):.4f}  Wilcoxon p={p_str} → {path}")

    pd.DataFrame({
        "pid":         [r["pid"]   for r in records],
        "label":       [r["label"] for r in records],
        "center_drop": center_drops,
        "periph_drop": periph_drops,
    }).to_csv(os.path.join(out_dir, f"center_periphery_{safe}.csv"), index=False)
    return center_drops, periph_drops


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5b – Center vs periphery aggregate (reads per-fold CSVs, no model needed)
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_center_periphery(out_dir, label_a, label_b, n_folds):
    """
    Read center_periphery_<label>.csv from each fold subdirectory,
    merge all folds, and produce a single summary figure comparing:
      - Group A: center_drop vs periph_drop  (small patch)
      - Group B: center_drop vs periph_drop  (large patch)

    Output
    ------
    average/center_periphery_aggregate.png
    average/center_periphery_aggregate.csv
    """
    safe_a = label_a.replace(" ", "_").replace("/", "")
    safe_b = label_b.replace(" ", "_").replace("/", "")

    dfs_a, dfs_b = [], []
    for fold_idx in range(1, n_folds + 1):
        fold_dir = os.path.join(out_dir, f"fold{fold_idx}")
        path_a   = os.path.join(fold_dir, f"center_periphery_{safe_a}.csv")
        path_b   = os.path.join(fold_dir, f"center_periphery_{safe_b}.csv")
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"  [5b] fold {fold_idx}: CSV not found, skipping.")
            continue
        df_a = pd.read_csv(path_a); df_a["fold"] = fold_idx; df_a["group"] = label_a
        df_b = pd.read_csv(path_b); df_b["fold"] = fold_idx; df_b["group"] = label_b
        dfs_a.append(df_a); dfs_b.append(df_b)

    if not dfs_a:
        print("  [5b] No fold CSVs found — skipping aggregate center/periphery plot.")
        return

    merged_a = pd.concat(dfs_a, ignore_index=True)
    merged_b = pd.concat(dfs_b, ignore_index=True)

    c_a = merged_a["center_drop"].values
    p_a = merged_a["periph_drop"].values
    c_b = merged_b["center_drop"].values
    p_b = merged_b["periph_drop"].values

    # Wilcoxon signed-rank: periph > center for each group
    _, wil_p_a = scipy_stats.wilcoxon(p_a, c_a, alternative="greater")
    _, wil_p_b = scipy_stats.wilcoxon(p_b, c_b, alternative="greater")
    wil_str_a = f"{wil_p_a:.3f}" if wil_p_a >= 0.001 else "<0.001"
    wil_str_b = f"{wil_p_b:.3f}" if wil_p_b >= 0.001 else "<0.001"

    # ── Figure: 2 subplots side by side ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    BOX_COLORS = {"center": "gold", "periph": "mediumpurple"}

    for ax, (c_vals, p_vals, lbl, wil_str) in zip(
        axes,
        [(c_a, p_a, label_a, wil_str_a),
         (c_b, p_b, label_b, wil_str_b)],
    ):
        bp = ax.boxplot(
            [c_vals, p_vals],
            tick_labels=["Center zeroed\n(tumor core)", "Periphery zeroed\n(peritumoral)"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor(BOX_COLORS["center"]); bp["boxes"][0].set_alpha(0.8)
        bp["boxes"][1].set_facecolor(BOX_COLORS["periph"]); bp["boxes"][1].set_alpha(0.8)
        for xpos, vals in enumerate([c_vals, p_vals], start=1):
            ax.scatter(
                [xpos + np.random.uniform(-0.08, 0.08) for _ in vals],
                vals, color="black", alpha=0.25, s=10, zorder=3,
            )
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_ylabel("Prob drop  (base − ablated)")
        ax.set_title(
            f"{lbl}  (n={len(c_vals)},  {len(set(merged_a['fold'] if lbl==label_a else merged_b['fold']))} folds)\n"
            f"center={c_vals.mean():.4f}   periph={p_vals.mean():.4f}\n"
            f"Wilcoxon(periph>center) p={wil_str}"
        )

    fig.suptitle("Center vs Periphery Ablation — aggregate across folds", y=1.02)
    plt.tight_layout()

    avg_dir  = os.path.join(out_dir, "average")
    os.makedirs(avg_dir, exist_ok=True)
    path_fig = os.path.join(avg_dir, "center_periphery_aggregate.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [5b] Center/periph aggregate → {path_fig}")
    print(f"       {label_a}: center={c_a.mean():.4f}  periph={p_a.mean():.4f}  Wilcoxon p={wil_str_a}")
    print(f"       {label_b}: center={c_b.mean():.4f}  periph={p_b.mean():.4f}  Wilcoxon p={wil_str_b}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    pd.concat([merged_a, merged_b], ignore_index=True).to_csv(
        os.path.join(avg_dir, "center_periphery_aggregate.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5c – Ring / offset-attention comparison (MIL-specific)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ring(records, mil_offsets):
    """
    For each patch position (sorted by distance from tumour centre), compute
    the mean and std of attention weights across all patients.

    Returns
    -------
    stats  : dict  {group_name: {"mean_attn": ndarray(N,), "std_attn": ndarray(N,), "n": int}}
             Arrays are already reordered by ascending distance.
    order  : ndarray(N,)  original indices sorted by distance
    dist_mm: ndarray(N,)  distance of each patch centre from tumour centre (mm)
    """
    offsets  = np.array(mil_offsets, dtype=float)
    sx, sy, sz = SPACING
    dist_all = np.sqrt((offsets[:, 0] * sx) ** 2 +
                       (offsets[:, 1] * sy) ** 2 +
                       (offsets[:, 2] * sz) ** 2)
    order = np.argsort(dist_all)

    group_map = {
        "All":           records,
        "Recurrence":    [r for r in records if r["label"] == 1],
        "No-recurrence": [r for r in records if r["label"] == 0],
    }
    stats = {}
    for gname, grecs in group_map.items():
        if not grecs:
            stats[gname] = {"mean_attn": None, "std_attn": None, "n": 0}
            continue
        mat = np.stack([r["attn"] for r in grecs])   # (n_patients, N)
        stats[gname] = {
            "mean_attn": mat.mean(axis=0)[order],
            "std_attn":  mat.std(axis=0)[order],
            "n":         len(grecs),
        }
    return stats, order, dist_all


def compare_ring_analysis(recs_a, recs_b, label_a, label_b,
                           offsets_a, offsets_b, out_dir,
                           fold_idx=None, suffix=""):
    """
    Per-fold ring analysis: plot mean attention vs patch distance for both groups.
    Also saves per-fold CSVs needed for aggregate_ring_analysis.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats_a, order_a, dist_a = _compute_ring(recs_a, offsets_a)
    stats_b, order_b, dist_b = _compute_ring(recs_b, offsets_b)

    N_a = len(order_a);  N_b = len(order_b)
    uniform_a = 1 / N_a; uniform_b = 1 / N_b

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    colors = {label_a: "steelblue", label_b: "tomato"}

    for ax, gname in zip(axes, GROUPS):
        sa = stats_a[gname]; sb = stats_b[gname]
        if sa["mean_attn"] is None and sb["mean_attn"] is None:
            ax.set_title(f"{gname} (n=0)"); continue

        x_a = np.arange(N_a); x_b = np.arange(N_b)
        w = 0.38

        if sa["mean_attn"] is not None:
            ax.bar(x_a - w / 2, sa["mean_attn"], width=w, yerr=sa["std_attn"],
                   capsize=3, color=colors[label_a], alpha=0.8, label=label_a)
        if sb["mean_attn"] is not None:
            ax.bar(x_b + w / 2, sb["mean_attn"], width=w, yerr=sb["std_attn"],
                   capsize=3, color=colors[label_b], alpha=0.8, label=label_b)

        ax.axhline(uniform_a, color=colors[label_a], ls="--", lw=1,
                   label=f"uniform ({label_a}) = {uniform_a:.3f}")
        ax.axhline(uniform_b, color=colors[label_b], ls="--", lw=1,
                   label=f"uniform ({label_b}) = {uniform_b:.3f}")

        n_a = sa["n"]; n_b = sb["n"]
        xlabels = [f"P{order_a[i]}\n{dist_a[order_a[i]]:.0f}mm" for i in range(N_a)]
        ax.set_xticks(x_a); ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel("Mean attention weight")
        ax.set_title(f"{gname}  ({label_a} n={n_a}, {label_b} n={n_b})")
        ax.legend(fontsize=7)

    title = "Offset-attention comparison  (attention vs patch distance from tumour)"
    if suffix: title += f"\n{suffix}"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "ring_analysis_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [R]  Ring analysis comparison → {path}")

    # ── Save per-fold CSV for aggregation ─────────────────────────────────────
    for stats, order, dist_all, lbl in [
        (stats_a, order_a, dist_a, label_a),
        (stats_b, order_b, dist_b, label_b),
    ]:
        rows = []
        for rank, orig_idx in enumerate(order):
            row = {
                "fold":       fold_idx if fold_idx is not None else 0,
                "group_label": lbl,
                "patch_rank": rank,
                "patch_idx":  int(orig_idx),
                "dist_mm":    float(dist_all[orig_idx]),
            }
            for gname in GROUPS:
                s = stats[gname]
                row[f"mean_attn_{gname}"] = (float(s["mean_attn"][rank])
                                              if s["mean_attn"] is not None else float("nan"))
                row[f"n_{gname}"]         = s["n"]
            rows.append(row)
        safe = lbl.replace(" ", "_").replace("/", "")
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"ring_analysis_{safe}.csv"), index=False)


def aggregate_ring_analysis(out_dir, label_a, label_b, n_folds):
    """
    Read ring_analysis_<label>.csv from each fold subdirectory,
    average mean_attn across folds, and produce a summary comparison figure.

    Output
    ------
    average/ring_analysis_aggregate.png
    average/ring_analysis_aggregate.csv
    """
    safe_a = label_a.replace(" ", "_").replace("/", "")
    safe_b = label_b.replace(" ", "_").replace("/", "")

    dfs = []
    for fold_idx in range(1, n_folds + 1):
        fold_dir = os.path.join(out_dir, f"fold{fold_idx}")
        for lbl, safe in [(label_a, safe_a), (label_b, safe_b)]:
            path = os.path.join(fold_dir, f"ring_analysis_{safe}.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
            else:
                print(f"  [Ra] fold {fold_idx} [{lbl}]: CSV not found, skipping.")

    if not dfs:
        print("  [Ra] No fold CSVs found — skipping aggregate ring analysis.")
        return

    all_df = pd.concat(dfs, ignore_index=True)

    avg_dir = os.path.join(out_dir, "average")
    os.makedirs(avg_dir, exist_ok=True)

    # ── Aggregate: mean±std across folds per (group_label, patch_rank, gname) ─
    agg_rows = []
    for gname in GROUPS:
        col = f"mean_attn_{gname}"
        for lbl in [label_a, label_b]:
            sub = all_df[all_df["group_label"] == lbl].copy()
            if sub.empty: continue
            grp = sub.groupby("patch_rank").agg(
                patch_idx =("patch_idx",  "first"),
                dist_mm   =("dist_mm",    "first"),
                mean_attn =(col,          "mean"),
                std_attn  =(col,          "std"),
                n_folds   =(col,          "count"),
            ).reset_index()
            grp["group_label"] = lbl
            grp["label_group"] = gname
            grp[f"n_{gname}"]  = sub.groupby("patch_rank")[f"n_{gname}"].mean().values
            agg_rows.append(grp)

    agg_df = pd.concat(agg_rows, ignore_index=True)
    agg_df.to_csv(os.path.join(avg_dir, "ring_analysis_aggregate.csv"), index=False)

    # ── Figure: 3 subplots (one per label group) ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    colors = {label_a: "steelblue", label_b: "tomato"}

    for ax, gname in zip(axes, GROUPS):
        col = f"mean_attn_{gname}"
        plotted = False
        for lbl in [label_a, label_b]:
            sub = agg_df[(agg_df["group_label"] == lbl) &
                         (agg_df["label_group"] == gname)].sort_values("patch_rank")
            if sub.empty: continue

            x        = sub["patch_rank"].values
            y        = sub["mean_attn"].values
            err      = sub["std_attn"].fillna(0).values
            dist_arr = sub["dist_mm"].values
            n_pts    = int(sub[f"n_{gname}"].mean()) if f"n_{gname}" in sub.columns else 0
            N        = len(x)
            uniform  = 1 / N

            w = 0.38
            offset = -w / 2 if lbl == label_a else w / 2
            ax.bar(x + offset, y, width=w, yerr=err, capsize=3,
                   color=colors[lbl], alpha=0.85, label=f"{lbl} (n≈{n_pts})")
            ax.axhline(uniform, color=colors[lbl], ls="--", lw=1,
                       label=f"uniform = {uniform:.3f}")
            if not plotted:
                xlabels = [f"P{int(sub.iloc[i]['patch_idx'])}\n{dist_arr[i]:.0f}mm"
                           for i in range(N)]
                ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7)
                plotted = True

        ax.set_ylabel("Mean attention weight  (avg across folds)")
        ax.set_title(gname)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Offset-attention comparison — {n_folds}-fold average\n"
        f"{label_a} vs {label_b}",
        y=1.02,
    )
    plt.tight_layout()
    path_fig = os.path.join(avg_dir, "ring_analysis_aggregate.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [Ra] Ring analysis aggregate → {path_fig}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 6 – Visual comparison (top non-uniform matched cases)
# ══════════════════════════════════════════════════════════════════════════════

def visualize_patch_comparison(recs_a, recs_b, label_a, label_b,
                                out_dir, n_cases=4, suffix=""):
    """Side-by-side CT/PET mid-slice of highest-attn patch for both groups."""
    for r in recs_a: r["_std"] = float(r["attn"].std())
    for r in recs_b: r["_std"] = float(r["attn"].std())

    da   = {r["pid"]: r for r in recs_a}
    db   = {r["pid"]: r for r in recs_b}
    pids = sorted(set(da) & set(db),
                  key=lambda p: -(da[p]["_std"] + db[p]["_std"]))
    pids = pids[:n_cases]
    if not pids:
        print("  [6] No shared patients for visualization — skipping.")
        return

    has_pet = da[pids[0]]["pet"] is not None
    n_rows  = 4 if has_pet else 2   # CT-A / PET-A / CT-B / PET-B  (or CT-A / CT-B)
    n_cols  = len(pids)

    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows + 0.8))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.06, wspace=0.04)

    for col, pid in enumerate(pids):
        for group_row, (rd, lbl) in enumerate([(da, label_a), (db, label_b)]):
            r       = rd[pid]
            top_idx = int(np.argmax(r["attn"]))

            # CT row
            ct_row = group_row * (2 if has_pet else 1)
            ct_vol = r["ct"][top_idx, 0].numpy()
            mid_z  = ct_vol.shape[0] // 2
            ax_ct  = fig.add_subplot(gs[ct_row, col])
            ax_ct.imshow(ct_vol[mid_z], cmap="gray",
                         vmin=ct_vol.min(), vmax=ct_vol.max())
            ax_ct.axis("off")
            if col == 0:
                ax_ct.set_ylabel(f"CT\n{lbl}", fontsize=8)
            if group_row == 0:
                lr = "R" if r["label"] == 1 else "NR"
                ax_ct.set_title(
                    f"{pid}\nstd={r['_std']:.3f} p={r['prob']:.2f} ({lr})",
                    fontsize=7,
                )

            # PET row
            if has_pet:
                pet_row = ct_row + 1
                pet_vol = r["pet"][top_idx, 0].numpy()
                ax_pet  = fig.add_subplot(gs[pet_row, col])
                ax_pet.imshow(pet_vol[mid_z], cmap="hot",
                              vmin=pet_vol.min(), vmax=pet_vol.max())
                ax_pet.axis("off")
                if col == 0:
                    ax_pet.set_ylabel(f"PET\n{lbl}", fontsize=8)

    title = f"Top-{len(pids)} non-uniform matched cases\nhighest-attn patch mid-slice"
    if suffix:
        title += f"\n{suffix}"
    fig.suptitle(title, y=1.01)
    path = os.path.join(out_dir, "patch_visual_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [6] Visual comparison → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-fold pair analysis
# ══════════════════════════════════════════════════════════════════════════════

def _analyse_fold_pair(model_a, recs_a, model_b, recs_b, device,
                        label_a, label_b, out_dir, args,
                        offsets_a=None, offsets_b=None,
                        suffix="", fold_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    compare_predictions(recs_a, recs_b, label_a, label_b, out_dir, suffix)
    compare_attn_std(recs_a, recs_b, label_a, label_b, out_dir, suffix)
    compare_top1_removal(model_a, recs_a, model_b, recs_b, device,
                          label_a, label_b, out_dir,
                          args.infer_batch_size, args.amp, suffix)
    center_periphery_ablation(model_a, recs_a, device, label_a, out_dir,
                               args.inner_frac, args.infer_batch_size,
                               args.amp, suffix)
    center_periphery_ablation(model_b, recs_b, device, label_b, out_dir,
                               args.inner_frac, args.infer_batch_size,
                               args.amp, suffix)
    # Ring / offset-attention analysis (MIL-specific)
    _offsets_a = offsets_a if offsets_a is not None else MIL_OFFSETS
    _offsets_b = offsets_b if offsets_b is not None else MIL_OFFSETS
    compare_ring_analysis(recs_a, recs_b, label_a, label_b,
                          _offsets_a, _offsets_b, out_dir,
                          fold_idx=fold_idx, suffix=suffix)
    visualize_patch_comparison(recs_a, recs_b, label_a, label_b,
                                out_dir, args.n_vis_cases, suffix)


# ══════════════════════════════════════════════════════════════════════════════
# Single-fold mode
# ══════════════════════════════════════════════════════════════════════════════

def run_single(args):
    label_a = args.label_a or f"patch_{int(args.patch_mm_a[0])}mm"
    label_b = args.label_b or f"patch_{int(args.patch_mm_b[0])}mm"

    ckpts_a = _collect_checkpoints(args.dir_a)
    ckpts_b = _collect_checkpoints(args.dir_b)
    fold    = args.fold
    ckpt_a  = ckpts_a[fold - 1]
    ckpt_b  = ckpts_b[fold - 1]

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_size == "small":
        _dla, _fus, _dose = DLA_CHANNELS_SMALL, FUSION_HIDDEN_SMALL, DOSE_CHANNELS_SMALL
    else:
        _dla, _fus, _dose = DLA_CHANNELS, FUSION_HIDDEN, DOSE_CHANNELS

    print(f"\n{'='*60}")
    print(f"  Patch size comparison  [single-fold, fold {fold}]")
    print(f"  {label_a}: {ckpt_a}")
    print(f"  {label_b}: {ckpt_b}")
    print(f"  Output : {args.out_dir}")
    print(f"{'='*60}\n")

    val_ids  = (_get_val_ids(args.csv, args.label_col, fold,
                             args.num_folds, args.seed)
                if args.val_only else None)
    offsets_a = _parse_offsets(args.mil_offsets_a)
    offsets_b = _parse_offsets(args.mil_offsets_b)

    ds_a = _make_dataset(args, args.patch_mm_a, offsets_a, val_ids)
    ds_b = _make_dataset(args, args.patch_mm_b, offsets_b, val_ids)

    model_a = _load_model(ckpt_a, device, args, _dla, _fus, _dose)
    model_b = _load_model(ckpt_b, device, args, _dla, _fus, _dose)

    print(f"  Exporting attention [{label_a}] ...")
    recs_a = _export_attention(model_a, ds_a, device,
                                args.export_batch_size, args.num_workers, args.amp)
    print(f"  Exporting attention [{label_b}] ...")
    recs_b = _export_attention(model_b, ds_b, device,
                                args.export_batch_size, args.num_workers, args.amp)

    auc_a = roc_auc_score([r["label"] for r in recs_a],
                           [r["prob"]  for r in recs_a])
    auc_b = roc_auc_score([r["label"] for r in recs_b],
                           [r["prob"]  for r in recs_b])
    plot_auc_comparison([auc_a], [auc_b], label_a, label_b, args.out_dir)

    _analyse_fold_pair(model_a, recs_a, model_b, recs_b, device,
                        label_a, label_b, args.out_dir, args,
                        offsets_a=offsets_a, offsets_b=offsets_b,
                        suffix=f"fold {fold}", fold_idx=fold)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-fold mode
# ══════════════════════════════════════════════════════════════════════════════

def run_multi(args):
    label_a = args.label_a or f"patch_{int(args.patch_mm_a[0])}mm"
    label_b = args.label_b or f"patch_{int(args.patch_mm_b[0])}mm"

    ckpts_a = _collect_checkpoints(args.dir_a)
    ckpts_b = _collect_checkpoints(args.dir_b)
    n_folds = min(len(ckpts_a), len(ckpts_b))

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_size == "small":
        _dla, _fus, _dose = DLA_CHANNELS_SMALL, FUSION_HIDDEN_SMALL, DOSE_CHANNELS_SMALL
    else:
        _dla, _fus, _dose = DLA_CHANNELS, FUSION_HIDDEN, DOSE_CHANNELS

    print(f"\n{'='*60}")
    print(f"  Patch size comparison  [multi-fold: {n_folds} folds]")
    print(f"  {label_a}: {args.dir_a}")
    print(f"  {label_b}: {args.dir_b}")
    print(f"  Output : {args.out_dir}")
    print(f"{'='*60}\n")

    offsets_a = _parse_offsets(args.mil_offsets_a)
    offsets_b = _parse_offsets(args.mil_offsets_b)

    auc_a_list, auc_b_list = [], []
    all_recs_a, all_recs_b = [], []   # merged across folds for aggregate plots

    for fold_idx in range(1, n_folds + 1):
        print(f"\n── Fold {fold_idx}/{n_folds}")
        ckpt_a   = ckpts_a[fold_idx - 1]
        ckpt_b   = ckpts_b[fold_idx - 1]
        fold_dir = os.path.join(args.out_dir, f"fold{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        val_ids = (_get_val_ids(args.csv, args.label_col, fold_idx,
                                args.num_folds, args.seed)
                   if args.val_only else None)

        ds_a = _make_dataset(args, args.patch_mm_a, offsets_a, val_ids)
        ds_b = _make_dataset(args, args.patch_mm_b, offsets_b, val_ids)

        model_a = _load_model(ckpt_a, device, args, _dla, _fus, _dose)
        model_b = _load_model(ckpt_b, device, args, _dla, _fus, _dose)

        recs_a = _export_attention(model_a, ds_a, device,
                                    args.export_batch_size, args.num_workers, args.amp)
        recs_b = _export_attention(model_b, ds_b, device,
                                    args.export_batch_size, args.num_workers, args.amp)

        auc_a = roc_auc_score([r["label"] for r in recs_a],
                               [r["prob"]  for r in recs_a])
        auc_b = roc_auc_score([r["label"] for r in recs_b],
                               [r["prob"]  for r in recs_b])
        auc_a_list.append(auc_a)
        auc_b_list.append(auc_b)
        all_recs_a.extend(recs_a)
        all_recs_b.extend(recs_b)

        _analyse_fold_pair(model_a, recs_a, model_b, recs_b, device,
                            label_a, label_b, fold_dir, args,
                            offsets_a=offsets_a, offsets_b=offsets_b,
                            suffix=f"fold {fold_idx}", fold_idx=fold_idx)

        del model_a, model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Aggregate analyses (no model needed) ──────────────────────────────────
    avg_dir = os.path.join(args.out_dir, "average")
    os.makedirs(avg_dir, exist_ok=True)
    print(f"\n── Aggregate analyses ({n_folds} folds merged) → {avg_dir}/")

    plot_auc_comparison(auc_a_list, auc_b_list, label_a, label_b, avg_dir)
    aggregate_center_periphery(args.out_dir, label_a, label_b, n_folds)
    aggregate_ring_analysis(args.out_dir, label_a, label_b, n_folds)
    compare_predictions(all_recs_a, all_recs_b, label_a, label_b, avg_dir,
                         suffix=f"all {n_folds} folds merged (n={len(all_recs_a)})")
    compare_attn_std(all_recs_a, all_recs_b, label_a, label_b, avg_dir,
                     suffix=f"all {n_folds} folds merged")
    visualize_patch_comparison(all_recs_a, all_recs_b, label_a, label_b,
                                avg_dir, args.n_vis_cases,
                                suffix=f"all {n_folds} folds merged")

    print(f"\nDone.")
    print(f"  Per-fold outputs : {args.out_dir}/fold{{1..{n_folds}}}/")
    print(f"  Aggregate output : {avg_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_offsets(offsets_arg):
    """Flat list [dx1,dy1,dz1, dx2,dy2,dz2,...] → list of (dx,dy,dz) tuples."""
    if offsets_arg is None:
        return MIL_OFFSETS
    flat = list(offsets_arg)
    if len(flat) % 3 != 0:
        raise ValueError("--mil_offsets_? must have a multiple-of-3 number of values")
    return [tuple(flat[i:i+3]) for i in range(0, len(flat), 3)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare MIL models trained with two different patch sizes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("mode", choices=["single", "multi"],
                   help="single: one fold pair; multi: all folds + aggregate.")
    # ── Model folders ──────────────────────────────────────────────────────────
    p.add_argument("--dir_a",   required=True,
                   help="Folder with fold*_best.pth for group A (small patch).")
    p.add_argument("--dir_b",   required=True,
                   help="Folder with fold*_best.pth for group B (large patch).")
    p.add_argument("--label_a", default=None,
                   help="Display name for group A (default: patch_<x>mm).")
    p.add_argument("--label_b", default=None,
                   help="Display name for group B (default: patch_<x>mm).")
    # ── Patch sizes ────────────────────────────────────────────────────────────
    p.add_argument("--patch_mm_a", type=float, nargs=3, metavar=("X","Y","Z"),
                   required=True, help="Patch physical size in mm for group A.")
    p.add_argument("--patch_mm_b", type=float, nargs=3, metavar=("X","Y","Z"),
                   required=True, help="Patch physical size in mm for group B.")
    p.add_argument("--mil_offsets_a", type=int, nargs="+", default=None,
                   help="Flat (dx dy dz …) offsets for group A; default: config MIL_OFFSETS.")
    p.add_argument("--mil_offsets_b", type=int, nargs="+", default=None,
                   help="Flat (dx dy dz …) offsets for group B; default: config MIL_OFFSETS.")
    # ── Data ───────────────────────────────────────────────────────────────────
    p.add_argument("--csv",             required=True)
    p.add_argument("--ct_dir",          required=True)
    p.add_argument("--pet_dir",         default=None)
    p.add_argument("--mask_dir",        default=None)
    p.add_argument("--label_col",       default="recurrence")
    p.add_argument("--ct_axis_order",   default="ZYX")
    p.add_argument("--pet_axis_order",  default="XYZ")
    p.add_argument("--ct_wl",   type=float, default=40)
    p.add_argument("--ct_ww",   type=float, default=400)
    p.add_argument("--pet_max", type=float, default=200)
    p.add_argument("--use_dose",        action="store_true")
    p.add_argument("--dose_dir",        default=None)
    p.add_argument("--dose_axis_order", default="ZYX")
    p.add_argument("--dose_max",        type=float, default=1.0)
    # ── Model ──────────────────────────────────────────────────────────────────
    p.add_argument("--model_size", default="base", choices=["base", "small"])
    # ── Val split ──────────────────────────────────────────────────────────────
    p.add_argument("--val_only",  action="store_true",
                   help="Evaluate on each fold's val patients only (recommended).")
    p.add_argument("--fold",      type=int, default=1,
                   help="Fold index for single mode (1-based, default 1).")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed",      type=int, default=42)
    # ── Ablation ───────────────────────────────────────────────────────────────
    p.add_argument("--inner_frac", type=float, default=0.5,
                   help="Fraction of patch dims defining the center region (default 0.5).")
    # ── Output & speed ─────────────────────────────────────────────────────────
    p.add_argument("--out_dir",           default="outputs/patch_size_comparison")
    p.add_argument("--amp",               action="store_true")
    p.add_argument("--infer_batch_size",  type=int, default=64)
    p.add_argument("--export_batch_size", type=int, default=4)
    p.add_argument("--num_workers",       type=int, default=4)
    p.add_argument("--n_vis_cases",       type=int, default=4,
                   help="Cases to visualize in patch comparison (default 4).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "single":
        run_single(args)
    else:
        run_multi(args)


if __name__ == "__main__":
    main()
