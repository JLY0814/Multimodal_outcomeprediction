"""
mil_analysis.py – Mechanistic analysis of MIL-attention models.

Four analyses
─────────────
1. Attention export   : per-patient per-patch weights → CSV
2. Ring analysis      : mean attention vs. physical distance from tumour centre
3. Top-k removal      : prediction drop when highest-weight patches are zeroed out
                        (compared to random-k removal as baseline)
4. Modality ablation  : CT vs. PET contribution per patch position
                        (zero out one modality for one patch, measure Δprob)

All analyses split results by label: All / Recurrence / No-recurrence.

Modes
─────
Single-model  (--checkpoint PATH):
    Analyse one checkpoint. Results go to --out_dir/.

Multi-model   (--checkpoints PATH1 PATH2 ...):
    Each model runs independently → out_dir/model_0/, model_1/, …
    Cross-model averages saved   → out_dir/average/
    With multiple GPUs, models are distributed across devices automatically.

Speed notes
───────────
• topk_removal / modality_ablation batch ALL variant inputs for one patient
  into a single GPU call instead of sequential per-variant passes.
  For N=7 patches, n_rand=10: 66 → 1 call (topk), 14 → 1 call (modality).
• Use --amp to enable float16 inference (~2× throughput on Ampere/Volta GPUs).
• Use --infer_batch_size to control GPU memory per call (default 128).
• Use --num_workers for parallel data loading in export_attention.
• Multi-GPU: in multi-model mode each model is placed on a different GPU
  (round-robin) so all GPUs run simultaneously.

Usage
─────
# single model
python mil_analysis.py --checkpoint fold1.pth --csv ... --ct_dir ...

# multi model (uses all available GPUs automatically)
python mil_analysis.py --checkpoints fold1.pth fold2.pth fold3.pth \\
    --csv ... --ct_dir ... --amp
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    MIL_PATCH_SIZE, MIL_OFFSETS, SPACING,
    DLA_CHANNELS, DLA_CHANNELS_SMALL,
    FUSION_HIDDEN, FUSION_HIDDEN_SMALL,
    DOSE_CHANNELS, DOSE_CHANNELS_SMALL,
    MIL_ATTN_DIM, DROPOUT,
)
from sklearn.model_selection import StratifiedKFold
from evaluate import load_model
from dataset import PatchDataset, parse_labels
from utils import set_seed


# ── Val-split helper ───────────────────────────────────────────────────────────

def get_val_ids(csv_path, label_col, fold_idx, num_folds=5, seed=42):
    """
    Reproduce the StratifiedKFold split used in main.py and return the
    Patient_ID list for the validation set of fold_idx (1-based).
    """
    df  = pd.read_csv(csv_path)
    df  = parse_labels(df, label_col)
    ids = df["Patient_ID"].astype(str).values
    lbs = df["label"].astype(int).values
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for i, (_, val_idx) in enumerate(skf.split(ids, lbs), start=1):
        if i == fold_idx:
            return ids[val_idx].tolist()
    raise ValueError(f"fold_idx={fold_idx} out of range for num_folds={num_folds}")


# ── Constants ──────────────────────────────────────────────────────────────────

GROUPS  = ["All", "Recurrence", "No-recurrence"]
PALETTE = {
    "All":           ("steelblue",  "gray"),
    "Recurrence":    ("tomato",     "salmon"),
    "No-recurrence": ("seagreen",   "mediumseagreen"),
}


# ── Attention hook ─────────────────────────────────────────────────────────────

class _AttnCapture:
    """Forward hook that stores GatedAttentionMIL attention weights."""
    def __init__(self): self.A = None
    def hook(self, module, inp, out):
        # GatedAttentionMIL.forward → (patient_emb, A);  A shape: (B, N, 1)
        self.A = out[1].detach().cpu()


def register_attn_hook(model):
    cap = _AttnCapture()
    for m in model.modules():
        if m.__class__.__name__ == "GatedAttentionMIL":
            handle = m.register_forward_hook(cap.hook)
            return cap, handle
    raise RuntimeError(
        "No GatedAttentionMIL found – was the checkpoint trained with --use_mil?"
    )


# ── Shared geometry helper ─────────────────────────────────────────────────────

def _patch_order():
    """Return (order, dist_mm, xlabels) with patches sorted by distance from centre."""
    offsets = np.array(MIL_OFFSETS, dtype=float)
    sx, sy, sz = SPACING
    dist_mm = np.sqrt((offsets[:, 0] * sx) ** 2 +
                      (offsets[:, 1] * sy) ** 2 +
                      (offsets[:, 2] * sz) ** 2)
    order   = np.argsort(dist_mm)
    dists   = dist_mm[order]
    xlabels = [f"P{order[i]}\n({dists[i]:.0f} mm)" for i in range(len(order))]
    return order, dist_mm, xlabels


# ── Label-split helpers ────────────────────────────────────────────────────────

def _split_by_label(records):
    return {
        "All":           records,
        "Recurrence":    [r for r in records if r["label"] == 1],
        "No-recurrence": [r for r in records if r["label"] == 0],
    }


def _bool_masks(records):
    labels = np.array([r["label"] for r in records])
    return {
        "All":           np.ones(len(records), dtype=bool),
        "Recurrence":    labels == 1,
        "No-recurrence": labels == 0,
    }


# ── Batched inference helper ───────────────────────────────────────────────────

@torch.no_grad()
def _batched_infer(model, device, ct_list, pet_list, dose_list,
                   infer_batch_size, use_amp):
    """
    Run model on a list of (N, 1, D, H, W) tensors in chunks.

    Stacks them into (chunk, N, 1, D, H, W) batches and returns
    a flat np.float32 array of sigmoid probabilities.

    This replaces sequential per-variant forward passes with a single
    (or a few) batched GPU calls — the core speedup for topk and modality.
    """
    all_probs = []
    n = len(ct_list)
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) \
              if (use_amp and device.type == "cuda") else torch.no_grad()

    for start in range(0, n, infer_batch_size):
        end = min(start + infer_batch_size, n)
        batch_ct   = torch.stack(ct_list[start:end]).to(device, non_blocking=True)
        batch_pet  = torch.stack(pet_list[start:end]).to(device, non_blocking=True)
        batch_dose = torch.stack(dose_list[start:end]).to(device, non_blocking=True)
        with amp_ctx:
            logits = model(batch_ct, batch_pet, batch_dose)
        all_probs.append(torch.sigmoid(logits).float().cpu().numpy())

    return np.concatenate(all_probs)   # (n,)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1 – Attention export
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_attention(model, dataset, device, out_dir,
                     batch_size=4, num_workers=4, use_amp=False,
                     use_bg_patches=False):
    """
    Run every patient through the model and collect per-patch attention weights.

    batch_size > 1 amortises DataLoader overhead; num_workers parallelises
    .npy loading. The attention hook handles B > 1 correctly (shape B, N, 1).

    When use_bg_patches=True the dataset returns a 5-tuple that includes
    patch_types (0=tumor / 1=background), which is stored in each record.

    Returns
    -------
    records : list[dict]  keys: pid, label, prob, attn (np.ndarray N),
                                ct / pet / dose  (cpu tensors, N×1×D×H×W),
                                patch_types (np.ndarray N, or None)
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        timeout=120 if num_workers > 0 else 0,
    )
    cap, handle = register_attn_hook(model)
    model.eval()
    records     = []
    patient_idx = 0

    amp_ctx = torch.autocast("cuda", dtype=torch.float16) \
              if (use_amp and device.type == "cuda") else torch.no_grad()

    for batch in tqdm(loader, desc="[1] Attention export"):
        if use_bg_patches:
            ct, pet, dose, labels, batch_patch_types = batch
        else:
            ct, pet, dose, labels = batch
            batch_patch_types = None

        with amp_ctx:
            logits = model(ct.to(device), pet.to(device), dose.to(device))
        probs = torch.sigmoid(logits).float().cpu()   # (B,)
        attns = cap.A[:, :, 0].cpu()                  # (B, N)

        for b in range(ct.shape[0]):
            pid = str(dataset.df.iloc[patient_idx]["Patient_ID"])
            records.append(dict(
                pid=pid, label=int(labels[b].item()),
                prob=float(probs[b]),
                attn=attns[b].numpy(),             # (N,)
                ct=ct[b].cpu(),                    # (N, 1, D, H, W)
                pet=pet[b].cpu(),
                dose=dose[b].cpu(),
                patch_types=(batch_patch_types[b].numpy()
                             if batch_patch_types is not None else None),
            ))
            patient_idx += 1

    handle.remove()

    if out_dir is not None:
        N    = records[0]["attn"].shape[0] if records else 0
        rows = []
        for r in records:
            row = {"pid": r["pid"], "label": r["label"], "prob": r["prob"]}
            for j in range(N):
                row[f"attn_{j}"] = float(r["attn"][j])
                if r["patch_types"] is not None:
                    row[f"type_{j}"] = int(r["patch_types"][j])   # 0=tumor,1=bg
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "attention_weights.csv")
        df.to_csv(csv_path, index=False)
        type_note = " (+ type_j columns: 0=tumor/1=bg)" if use_bg_patches else ""
        print(f"  → {csv_path}  ({len(df)} patients, {N} patches each{type_note})")

    return records


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2 – Ring analysis
# ══════════════════════════════════════════════════════════════════════════════

def _compute_ring_stats(records):
    order, _, _ = _patch_order()
    stats = {}
    for gname, grecs in _split_by_label(records).items():
        if len(grecs) == 0:
            stats[gname] = {"mean_attn": None, "std_attn": None, "n": 0}
        else:
            attn = np.stack([r["attn"] for r in grecs])[:, order]   # (P, N)
            stats[gname] = {
                "mean_attn": attn.mean(axis=0),
                "std_attn":  attn.std(axis=0),
                "n":         len(grecs),
            }
    return stats


def _plot_ring(stats, out_dir, title_suffix=""):
    order, _, xlabels = _patch_order()
    N       = len(order)
    uniform = 1 / N
    colors  = {"All": "steelblue", "Recurrence": "tomato", "No-recurrence": "seagreen"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    csv_rows  = []

    for ax, gname in zip(axes, GROUPS):
        s = stats[gname]
        if s["n"] == 0:
            ax.set_title(f"{gname} (n=0)"); continue
        x = np.arange(N)
        ax.bar(x, s["mean_attn"], yerr=s["std_attn"], capsize=4,
               color=colors[gname], alpha=0.85)
        ax.axhline(uniform, ls="--", color="black", lw=1.2,
                   label=f"Uniform = {uniform:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"{gname}  (n={s['n']})")
        ax.set_ylabel("Mean attention weight")
        ax.legend(fontsize=7)
        for i in range(N):
            csv_rows.append({
                "group": gname, "patch_idx": int(order[i]),
                "mean_attn": float(s["mean_attn"][i]),
                "std_attn":  float(s["std_attn"][i]),
            })

    title = "Ring analysis: attention vs. distance from tumour centre"
    if title_suffix:
        title += f"  [{title_suffix}]"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "ring_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[2] Ring analysis → {path}")
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "ring_analysis.csv"), index=False)


def ring_analysis(records, out_dir, title_suffix=""):
    stats = _compute_ring_stats(records)
    _plot_ring(stats, out_dir, title_suffix)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 3 – Top-k patch removal
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _compute_topk_stats(model, records, device, n_rand_trials=10,
                         infer_batch_size=128, use_amp=False):
    """
    Key speedup: for each patient, build ALL (N-1)*(1+n_rand) variant inputs
    at once and run them in a single batched GPU call instead of sequential
    per-variant calls.

    N is read from records[0]["attn"].shape[0] (actual bag size), so this works
    for both standard MIL (N=7) and BG-patch mode (N=2*n_slices).
    """
    model.eval()
    # Use actual patch count from records (supports both standard MIL and BG-patch mode)
    N        = records[0]["attn"].shape[0] if records else len(MIL_OFFSETS)
    k_values = list(range(1, N))
    rng      = np.random.default_rng(42)

    # {k: [(label, top_drop, rand_drop), ...]}
    per_patient = {k: [] for k in k_values}

    for r in tqdm(records, desc="[3] Top-k removal"):
        prob_base = r["prob"]
        label     = r["label"]
        top_order = np.argsort(r["attn"])[::-1]
        ct0  = r["ct"]    # (N, 1, D, H, W) — no unsqueeze, stack for batch
        p0   = r["pet"]
        d0   = r["dose"]

        # ── Build all variant tensors for this patient ─────────────────────
        var_ct, var_pet, var_dose = [], [], []
        # meta: (k, is_top_k)
        meta = []

        for k in k_values:
            # Top-k variant
            ct_m = ct0.clone(); pm = p0.clone(); dm = d0.clone()
            for idx in top_order[:k]:
                ct_m[idx] = 0.; pm[idx] = 0.; dm[idx] = 0.
            var_ct.append(ct_m); var_pet.append(pm); var_dose.append(dm)
            meta.append((k, True))

            # Random-k variants
            for _ in range(n_rand_trials):
                ri = rng.choice(N, size=k, replace=False)
                ct_r = ct0.clone(); pr = p0.clone(); dr = d0.clone()
                for idx in ri:
                    ct_r[idx] = 0.; pr[idx] = 0.; dr[idx] = 0.
                var_ct.append(ct_r); var_pet.append(pr); var_dose.append(dr)
                meta.append((k, False))

        # ── Single batched inference call ──────────────────────────────────
        all_probs = _batched_infer(model, device, var_ct, var_pet, var_dose,
                                   infer_batch_size, use_amp)

        # ── Parse results ──────────────────────────────────────────────────
        rand_accum = {k: [] for k in k_values}
        top_drop   = {}
        for vi, (k, is_top) in enumerate(meta):
            drop = prob_base - all_probs[vi]
            if is_top:
                top_drop[k] = drop
            else:
                rand_accum[k].append(drop)

        for k in k_values:
            per_patient[k].append((label, top_drop[k],
                                   float(np.mean(rand_accum[k]))))

    # ── Aggregate by group ─────────────────────────────────────────────────
    group_filter = {"All": None, "Recurrence": 1, "No-recurrence": 0}
    stats = {}
    for gname, lf in group_filter.items():
        top_means, top_stds, rand_means, rand_stds = [], [], [], []
        for k in k_values:
            entries = per_patient[k] if lf is None \
                      else [e for e in per_patient[k] if e[0] == lf]
            tops  = [e[1] for e in entries]
            rands = [e[2] for e in entries]
            top_means.append(np.mean(tops)   if tops  else float("nan"))
            top_stds.append( np.std(tops)    if tops  else float("nan"))
            rand_means.append(np.mean(rands) if rands else float("nan"))
            rand_stds.append( np.std(rands)  if rands else float("nan"))
        n = len(per_patient[k_values[0]]) if lf is None else \
            sum(1 for e in per_patient[k_values[0]] if e[0] == lf)
        stats[gname] = dict(
            top_means=top_means, top_stds=top_stds,
            rand_means=rand_means, rand_stds=rand_stds, n=n,
        )
    return stats, k_values


def _plot_topk(stats, k_values, out_dir, n_rand_trials=10, title_suffix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)

    for ax, gname in zip(axes, GROUPS):
        s = stats[gname]
        col_top, col_rand = PALETTE[gname]
        if s["n"] == 0:
            ax.set_title(f"{gname} (n=0)"); continue
        ax.errorbar(k_values, s["top_means"],  yerr=s["top_stds"],
                    marker="o", capsize=4,
                    label="Top-k (highest attention)", color=col_top)
        ax.errorbar(k_values, s["rand_means"], yerr=s["rand_stds"],
                    marker="s", capsize=4, ls="--",
                    label=f"Random-k (avg {n_rand_trials} trials)", color=col_rand)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("k  (patches zeroed out)")
        ax.set_ylabel("Mean probability drop  (base − modified)")
        ax.set_title(f"{gname}  (n={s['n']})")
        ax.set_xticks(k_values)
        ax.legend(fontsize=7)

    title = "Top-k patch removal: functional validation"
    if title_suffix:
        title += f"  [{title_suffix}]"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "topk_removal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[3] Top-k removal → {path}")

    csv_rows = []
    for gname, s in stats.items():
        for i, k in enumerate(k_values):
            csv_rows.append({
                "group": gname, "k": k,
                "top_mean_drop":  s["top_means"][i],  "top_std_drop":  s["top_stds"][i],
                "rand_mean_drop": s["rand_means"][i], "rand_std_drop": s["rand_stds"][i],
            })
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "topk_removal.csv"), index=False)


def topk_removal(model, records, device, out_dir,
                 n_rand_trials=10, infer_batch_size=128, use_amp=False,
                 title_suffix=""):
    stats, k_values = _compute_topk_stats(model, records, device,
                                           n_rand_trials, infer_batch_size, use_amp)
    _plot_topk(stats, k_values, out_dir, n_rand_trials, title_suffix)
    return stats, k_values


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 4 – Modality ablation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _compute_modality_stats(model, records, device,
                             infer_batch_size=128, use_amp=False):
    """
    Key speedup: for each patient, build all 2*N ablation inputs at once
    (N CT-ablation + N PET-ablation variants) and run a single batched call.

    N is read from records[0]["attn"].shape[0] (actual bag size), so this works
    for both standard MIL (N=7) and BG-patch mode (N=2*n_slices).
    """
    model.eval()
    # Use actual patch count from records (supports standard MIL and BG-patch mode)
    N = records[0]["attn"].shape[0] if records else len(MIL_OFFSETS)

    ct_contrib  = np.zeros((len(records), N))
    pet_contrib = np.zeros((len(records), N))

    for i, r in enumerate(tqdm(records, desc="[4] Modality ablation")):
        prob_base = r["prob"]
        ct0 = r["ct"]    # (N, 1, D, H, W)
        p0  = r["pet"]
        d0  = r["dose"]

        # ── Build all 2*N ablation variants ───────────────────────────────
        var_ct, var_pet, var_dose = [], [], []
        for j in range(N):
            # CT ablation at patch j  (keep pet/dose original)
            ct_no = ct0.clone(); ct_no[j] = 0.
            var_ct.append(ct_no); var_pet.append(p0); var_dose.append(d0)
        for j in range(N):
            # PET ablation at patch j  (keep ct/dose original)
            pet_no = p0.clone(); pet_no[j] = 0.
            var_ct.append(ct0); var_pet.append(pet_no); var_dose.append(d0)

        # ── Single batched inference call ──────────────────────────────────
        all_probs = _batched_infer(model, device, var_ct, var_pet, var_dose,
                                   infer_batch_size, use_amp)

        for j in range(N):
            ct_contrib[i, j]  = prob_base - all_probs[j]
            pet_contrib[i, j] = prob_base - all_probs[N + j]

    # In BG-patch mode N != len(MIL_OFFSETS), so geometric ordering is invalid.
    # Fall back to simple sequential order with no distance info.
    if N == len(MIL_OFFSETS):
        order, dist_mm, _ = _patch_order()
    else:
        order   = np.arange(N)
        dist_mm = np.zeros(N)

    masks = _bool_masks(records)

    stats = {}
    for gname, mask in masks.items():
        if mask.sum() == 0:
            stats[gname] = {"ct_mean": None, "ct_std": None,
                            "pet_mean": None, "pet_std": None, "n": 0}
        else:
            ct_ord  = ct_contrib[mask][:,  order]
            pet_ord = pet_contrib[mask][:, order]
            stats[gname] = {
                "ct_mean":  ct_ord.mean(axis=0),
                "ct_std":   ct_ord.std(axis=0),
                "pet_mean": pet_ord.mean(axis=0),
                "pet_std":  pet_ord.std(axis=0),
                "n":        int(mask.sum()),
            }
    return stats, order, dist_mm


def _plot_modality(stats, order, dist_mm, out_dir, title_suffix=""):
    N = len(order)
    # If dist_mm is all zeros (BG-patch mode), show patch index + tumor/bg type
    if np.all(dist_mm == 0):
        n_slices = N // 2
        def _bg_label(patch_idx):
            sl   = patch_idx // 2
            kind = "T" if patch_idx % 2 == 0 else "B"
            return f"S{sl}-{kind}"
        xlabels = [_bg_label(order[j]) for j in range(N)]
    else:
        xlabels = [f"P{order[j]}\n({dist_mm[order[j]]:.0f} mm)" for j in range(N)]
    x = np.arange(N); w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(21, 4), sharey=True)
    csv_rows  = []

    for ax, gname in zip(axes, GROUPS):
        s = stats[gname]
        if s["n"] == 0:
            ax.set_title(f"{gname} (n=0)"); continue
        ax.bar(x - w/2, s["ct_mean"],  width=w, yerr=s["ct_std"],  capsize=3,
               label="CT ablation",  color="steelblue", alpha=0.85)
        ax.bar(x + w/2, s["pet_mean"], width=w, yerr=s["pet_std"], capsize=3,
               label="PET ablation", color="tomato",    alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Mean prob drop when modality ablated")
        ax.set_title(f"{gname}  (n={s['n']})")
        ax.legend(fontsize=7)
        for j in range(N):
            oj = order[j]
            csv_rows.append({
                "group": gname, "patch_idx": int(oj),
                "dist_mm": float(dist_mm[oj]),
                "ct_mean_drop":  float(s["ct_mean"][j]),
                "ct_std_drop":   float(s["ct_std"][j]),
                "pet_mean_drop": float(s["pet_mean"][j]),
                "pet_std_drop":  float(s["pet_std"][j]),
            })

    title = "Modality contribution per patch position"
    if title_suffix:
        title += f"  [{title_suffix}]"
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "modality_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[4] Modality ablation → {path}")
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "modality_ablation.csv"), index=False)


def modality_ablation(model, records, device, out_dir,
                       infer_batch_size=128, use_amp=False, title_suffix=""):
    stats, order, dist_mm = _compute_modality_stats(model, records, device,
                                                      infer_batch_size, use_amp)
    _plot_modality(stats, order, dist_mm, out_dir, title_suffix)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Cross-model averaging (multi-model mode only)
# ══════════════════════════════════════════════════════════════════════════════

def _avg_attention(all_records_list, out_dir):
    N          = all_records_list[0][0]["attn"].shape[0] if all_records_list and all_records_list[0] else len(MIL_OFFSETS)
    n_models   = len(all_records_list)
    n_patients = len(all_records_list[0])
    rows = []
    for i in range(n_patients):
        r0       = all_records_list[0][i]
        avg_attn = np.mean([all_records_list[m][i]["attn"] for m in range(n_models)], axis=0)
        avg_prob = float(np.mean([all_records_list[m][i]["prob"] for m in range(n_models)]))
        row = {"pid": r0["pid"], "label": r0["label"], "prob_avg": avg_prob}
        for j in range(N):
            row[f"attn_{j}_avg"] = float(avg_attn[j])
        rows.append(row)
    csv_path = os.path.join(out_dir, "attention_weights_avg.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[avg][1] Attention weights → {csv_path}  ({n_patients} pts, {n_models} models)")


def _avg_ring(all_ring_stats, out_dir, n_models):
    order, _, xlabels = _patch_order()
    N       = len(order)
    uniform = 1 / N
    colors  = {"All": "steelblue", "Recurrence": "tomato", "No-recurrence": "seagreen"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    csv_rows  = []

    for ax, gname in zip(axes, GROUPS):
        model_means = [s[gname]["mean_attn"] for s in all_ring_stats
                       if s[gname]["mean_attn"] is not None]
        if not model_means:
            ax.set_title(f"{gname} (n=0)"); continue
        arr      = np.stack(model_means)
        avg_mean = arr.mean(axis=0)
        avg_std  = arr.std(axis=0)
        n_pts    = all_ring_stats[0][gname]["n"]

        x = np.arange(N)
        ax.bar(x, avg_mean, yerr=avg_std, capsize=4, color=colors[gname], alpha=0.85)
        ax.axhline(uniform, ls="--", color="black", lw=1.2,
                   label=f"Uniform = {uniform:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"{gname}  (n={n_pts},  {n_models} models)")
        ax.set_ylabel("Mean attention (avg across models)")
        ax.legend(fontsize=7)
        for i in range(N):
            csv_rows.append({
                "group": gname, "patch_idx": int(order[i]),
                "mean_attn_avg":     float(avg_mean[i]),
                "std_across_models": float(avg_std[i]),
            })

    fig.suptitle(f"Ring analysis  (averaged over {n_models} models)", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "ring_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[avg][2] Ring analysis → {path}")
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "ring_analysis.csv"), index=False)


def _avg_topk(all_topk_stats, k_values, out_dir, n_models, n_rand_trials=10):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    csv_rows  = []

    for ax, gname in zip(axes, GROUPS):
        col_top, col_rand = PALETTE[gname]
        n = all_topk_stats[0][gname]["n"]
        if n == 0:
            ax.set_title(f"{gname} (n=0)"); continue

        top_arr  = np.array([s[gname]["top_means"]  for s in all_topk_stats])
        rand_arr = np.array([s[gname]["rand_means"] for s in all_topk_stats])
        avg_top  = top_arr.mean(axis=0);  std_top  = top_arr.std(axis=0)
        avg_rand = rand_arr.mean(axis=0); std_rand = rand_arr.std(axis=0)

        ax.errorbar(k_values, avg_top,  yerr=std_top,
                    marker="o", capsize=4,
                    label="Top-k (highest attention)", color=col_top)
        ax.errorbar(k_values, avg_rand, yerr=std_rand,
                    marker="s", capsize=4, ls="--",
                    label=f"Random-k (avg {n_rand_trials} trials)", color=col_rand)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("k  (patches zeroed out)")
        ax.set_ylabel("Mean probability drop  (base − modified)")
        ax.set_title(f"{gname}  (n={n},  {n_models} models)")
        ax.set_xticks(k_values)
        ax.legend(fontsize=7)

        for i, k in enumerate(k_values):
            csv_rows.append({
                "group": gname, "k": k,
                "top_mean_drop_avg":      float(avg_top[i]),
                "top_std_across_models":  float(std_top[i]),
                "rand_mean_drop_avg":     float(avg_rand[i]),
                "rand_std_across_models": float(std_rand[i]),
            })

    fig.suptitle(f"Top-k patch removal  (averaged over {n_models} models)", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "topk_removal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[avg][3] Top-k removal → {path}")
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "topk_removal.csv"), index=False)


def _avg_modality(all_mod_stats, out_dir, n_models):
    # Detect actual N from data (BG-patch mode has N != len(MIL_OFFSETS))
    _sample_ct = next((s[g]["ct_mean"] for s in all_mod_stats
                       for g in GROUPS if s[g]["ct_mean"] is not None), None)
    N_actual = len(_sample_ct) if _sample_ct is not None else len(MIL_OFFSETS)

    if N_actual == len(MIL_OFFSETS):
        order, dist_mm, _ = _patch_order()
        xlabels = [f"P{order[j]}\n({dist_mm[order[j]]:.0f} mm)" for j in range(N_actual)]
    else:
        order   = np.arange(N_actual)
        dist_mm = np.zeros(N_actual)
        n_slices = N_actual // 2
        def _bg_lbl(idx): return f"S{idx//2}-{'T' if idx%2==0 else 'B'}"
        xlabels = [_bg_lbl(order[j]) for j in range(N_actual)]

    N       = N_actual
    x = np.arange(N); w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(21, 4), sharey=True)
    csv_rows  = []

    for ax, gname in zip(axes, GROUPS):
        ct_arrs  = [s[gname]["ct_mean"]  for s in all_mod_stats if s[gname]["ct_mean"]  is not None]
        pet_arrs = [s[gname]["pet_mean"] for s in all_mod_stats if s[gname]["pet_mean"] is not None]
        n = all_mod_stats[0][gname]["n"]
        if n == 0 or not ct_arrs:
            ax.set_title(f"{gname} (n=0)"); continue

        ct_mat  = np.stack(ct_arrs);  pet_mat = np.stack(pet_arrs)
        avg_ct  = ct_mat.mean(axis=0);  std_ct  = ct_mat.std(axis=0)
        avg_pet = pet_mat.mean(axis=0); std_pet = pet_mat.std(axis=0)

        ax.bar(x - w/2, avg_ct,  width=w, yerr=std_ct,  capsize=3,
               label="CT ablation",  color="steelblue", alpha=0.85)
        ax.bar(x + w/2, avg_pet, width=w, yerr=std_pet, capsize=3,
               label="PET ablation", color="tomato",    alpha=0.85)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Mean prob drop when modality ablated")
        ax.set_title(f"{gname}  (n={n},  {n_models} models)")
        ax.legend(fontsize=7)

        for j in range(N):
            oj = order[j]
            csv_rows.append({
                "group": gname, "patch_idx": int(oj),
                "dist_mm": float(dist_mm[oj]),
                "ct_mean_drop_avg":      float(avg_ct[j]),
                "ct_std_across_models":  float(std_ct[j]),
                "pet_mean_drop_avg":     float(avg_pet[j]),
                "pet_std_across_models": float(std_pet[j]),
            })

    fig.suptitle(f"Modality contribution  (averaged over {n_models} models)", y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "modality_ablation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[avg][4] Modality ablation → {path}")
    pd.DataFrame(csv_rows).to_csv(os.path.join(out_dir, "modality_ablation.csv"), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 5 – Tumor vs Background attention (BG-patch mode only)
# ══════════════════════════════════════════════════════════════════════════════

def tumor_vs_bg_analysis(records, out_dir, title_suffix=""):
    """
    Compare mean attention of tumor patches (type=0) vs background patches
    (type=1) for models trained with --use_bg_patches.

    Produces
    --------
    tumor_vs_bg_attention.png   : box plots per label group (tumor vs bg)
    attn_ratio_vs_prob.png      : scatter of (tumor/bg) ratio vs prediction prob
    tumor_vs_bg_stats.csv       : per-group Wilcoxon test results
    tumor_vs_bg_per_patient.csv : per-patient mean_tumor_attn, mean_bg_attn, ratio
    """
    from scipy import stats as scipy_stats

    # ── Per-patient metrics ───────────────────────────────────────────────────
    for r in records:
        pt = r.get("patch_types")
        if pt is None:
            r["mean_tumor_attn"] = float("nan")
            r["mean_bg_attn"]    = float("nan")
            r["attn_ratio"]      = float("nan")
            continue
        attn   = r["attn"]
        t_mask = (pt == 0)
        b_mask = (pt == 1)
        r["mean_tumor_attn"] = float(attn[t_mask].mean()) if t_mask.any() else float("nan")
        r["mean_bg_attn"]    = float(attn[b_mask].mean()) if b_mask.any() else float("nan")
        denom = r["mean_bg_attn"] if (not np.isnan(r["mean_bg_attn"]) and
                                       r["mean_bg_attn"] > 1e-9) else 1e-9
        r["attn_ratio"] = r["mean_tumor_attn"] / denom

    valid = [r for r in records if not np.isnan(r.get("mean_tumor_attn", float("nan")))]
    if not valid:
        print("  [BG][5] No records with patch_types – skipping tumor-vs-bg analysis.")
        return {}

    # ── Diagnose all-zero patch_types (fallback triggered) ───────────────────
    n_with_bg = sum(1 for r in valid
                    if not np.isnan(r.get("mean_bg_attn", float("nan"))))
    if n_with_bg == 0:
        print("  [BG][5] WARNING: bg_mean is nan for ALL patients.")
        print("          This usually means _make_patches_slice_bg fell back to")
        print("          offset-based patches (patch_types all 0) because no mask")
        print("          was found. Check that --mask_dir is correctly set and")
        print("          mask files exist as {Patient_ID}_MTV_Cervix.npy.")
        n_pt_none = sum(1 for r in valid if r.get("patch_types") is None)
        if n_pt_none > 0:
            print(f"          ({n_pt_none}/{len(valid)} records have patch_types=None"
                  " – export_attention may not be receiving patch_types from loader.)")
        else:
            sample_types = valid[0].get("patch_types", [])
            print(f"          Sample patch_types for first patient: {sample_types}"
                  f"  (unique values: {np.unique(sample_types).tolist()})")

    groups    = _split_by_label(valid)
    stats_out = {}

    # ── Figure 1: box plot – tumor vs bg attention per label group ────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    BOX_COLORS = ["steelblue", "salmon"]   # tumor, background

    for ax, (gname, grecs) in zip(axes, groups.items()):
        grecs_v = [r for r in grecs
                   if not np.isnan(r.get("mean_tumor_attn", float("nan")))]
        if len(grecs_v) < 2:
            ax.set_title(f"{gname}\n(n<2, skip)"); continue

        t_vals = [r["mean_tumor_attn"] for r in grecs_v]
        b_vals = [r["mean_bg_attn"]    for r in grecs_v]

        bp = ax.boxplot([t_vals, b_vals],
                        tick_labels=["Tumor\n(type=0)", "Background\n(type=1)"],
                        patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp["boxes"], BOX_COLORS):
            patch.set_facecolor(col); patch.set_alpha(0.7)

        # overlay individual points
        for xpos, vals in enumerate([t_vals, b_vals], start=1):
            ax.scatter([xpos + np.random.uniform(-0.08, 0.08)
                        for _ in vals], vals,
                       color="black", alpha=0.4, s=15, zorder=3)

        # Wilcoxon signed-rank: tumor > bg?
        try:
            stat, pval = scipy_stats.wilcoxon(t_vals, b_vals, alternative="greater")
        except ValueError:
            stat, pval = float("nan"), float("nan")

        p_str = f"{pval:.3f}" if not np.isnan(pval) else "N/A"
        ax.set_title(f"{gname}  (n={len(grecs_v)})\n"
                     f"Wilcoxon p={p_str} (tumor>bg?)")
        ax.set_ylabel("Mean attention weight")

        stats_out[gname] = dict(
            n             = len(grecs_v),
            tumor_mean    = float(np.mean(t_vals)),
            tumor_std     = float(np.std(t_vals)),
            bg_mean       = float(np.mean(b_vals)),
            bg_std        = float(np.std(b_vals)),
            wilcoxon_stat = float(stat) if not np.isnan(stat) else None,
            wilcoxon_p    = float(pval) if not np.isnan(pval) else None,
        )

    suptitle = "Tumor vs Background Attention"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    fig.suptitle(suptitle)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "tumor_vs_bg_attention.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [BG][5a] Tumor vs bg box plot → {path1}")

    # ── Figure 2: attn_ratio vs prediction probability ────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    COLOR_MAP = {"Recurrence": "tomato", "No-recurrence": "steelblue"}
    for gname, grecs_g in groups.items():
        if gname == "All":
            continue
        grecs_v = [r for r in grecs_g
                   if not np.isnan(r.get("attn_ratio", float("nan")))]
        ratios = [r["attn_ratio"] for r in grecs_v]
        probs  = [r["prob"]       for r in grecs_v]
        ax.scatter(ratios, probs, alpha=0.75, s=40, label=f"{gname} (n={len(grecs_v)})",
                   color=COLOR_MAP.get(gname, "gray"))

    ax.axvline(1.0, color="gray", linestyle="--", lw=1.2, label="ratio = 1")
    ax.set_xlabel("Attention ratio  (tumor / background)")
    ax.set_ylabel("Prediction probability")
    ax.set_title("Attn ratio vs. prediction" +
                 (f"\n{title_suffix}" if title_suffix else ""))
    ax.legend(fontsize=9)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "attn_ratio_vs_prob.png")
    plt.savefig(path2, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [BG][5b] Attn ratio vs prob  → {path2}")

    # ── CSVs ─────────────────────────────────────────────────────────────────
    pd.DataFrame([{"group": g, **v} for g, v in stats_out.items()]).to_csv(
        os.path.join(out_dir, "tumor_vs_bg_stats.csv"), index=False)
    pd.DataFrame([
        {"pid": r["pid"], "label": r["label"], "prob": r["prob"],
         "mean_tumor_attn": r.get("mean_tumor_attn"),
         "mean_bg_attn":    r.get("mean_bg_attn"),
         "attn_ratio":      r.get("attn_ratio")}
        for r in valid
    ]).to_csv(os.path.join(out_dir, "tumor_vs_bg_per_patient.csv"), index=False)

    # Print summary
    if "All" in stats_out:
        s = stats_out["All"]
        print(f"  [BG][5] Summary (All patients, n={s['n']}):")
        bg_m = s['bg_mean']
        bg_s = s['bg_std']
        bg_str = (f"{bg_m:.4f} ± {bg_s:.4f}"
                  if (bg_m is not None and not np.isnan(bg_m)) else "nan (no bg patches)")
        print(f"           tumor mean attn = {s['tumor_mean']:.4f} ± {s['tumor_std']:.4f}")
        print(f"           bg    mean attn = {bg_str}")
        p = s['wilcoxon_p']
        if p is not None:
            sig = "  ✓ significant" if p < 0.05 else "  (n.s.)"
            print(f"           Wilcoxon (tumor>bg): p = {p:.4f}{sig}")
        else:
            print(f"           Wilcoxon (tumor>bg): N/A")

    return stats_out


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 6 – Non-uniform attention case selection
# ══════════════════════════════════════════════════════════════════════════════

def nonuniform_attention_analysis(model, records, device, out_dir,
                                   infer_batch_size=128, use_amp=False,
                                   n_top_cases=6, n_std_group=10,
                                   title_suffix="",
                                   normalize_by_patient_baseline=True):
    """
    Select cases with significantly non-uniform attention and run three analyses:

    (a) Correlation: per-patient attn_std vs prediction confidence |prob − 0.5|
        Hypothesis: more focused attention → more confident prediction.

    (b) Top-1 removal: compare prob drop between high-std and low-std groups.
        Hypothesis: removing the top patch hurts high-std cases more,
        proving attention concentration is mechanistically meaningful.

    (c) Visual verification: CT/PET mid-slice of the highest-attention patch
        for the top-N most non-uniform cases.

    Outputs
    -------
    nonuniform_attn_std_vs_confidence.png
    nonuniform_topk_removal_comparison.png
    nonuniform_top_cases_visualization.png
    nonuniform_analysis.csv
    """
    import matplotlib.gridspec as gridspec
    from scipy import stats as scipy_stats
    from matplotlib.lines import Line2D

    os.makedirs(out_dir, exist_ok=True)

    # ── Per-patient attention statistics ──────────────────────────────────────
    for r in records:
        r["attn_std"]   = float(r["attn"].std())
        r["confidence"] = abs(float(r["prob"]) - 0.5)

    sorted_recs = sorted(records, key=lambda r: r["attn_std"], reverse=True)
    n       = len(sorted_recs)
    n_third = max(1, n // 3)
    high_std = sorted_recs[:n_third]
    low_std  = sorted_recs[-n_third:]

    attn_stds   = np.array([r["attn_std"]   for r in records])
    confidences = np.array([r["confidence"] for r in records])
    labels      = np.array([r["label"]      for r in records])

    thresh_high = high_std[-1]["attn_std"]
    thresh_low  = low_std[0]["attn_std"]

    print(f"\n  [NUA] Non-uniform attention analysis  (n={n})")
    print(f"         High-std group : top {n_third} cases  (std ≥ {thresh_high:.4f})")
    print(f"         Low-std  group : bot {n_third} cases  (std ≤ {thresh_low:.4f})")

    # ── (a) Attn std vs confidence ────────────────────────────────────────────
    r_val, p_val = scipy_stats.pearsonr(attn_stds, confidences)
    m, b = np.polyfit(attn_stds, confidences, 1)
    x_line = np.linspace(attn_stds.min(), attn_stds.max(), 100)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    colors = ["tomato" if l == 1 else "steelblue" for l in labels]
    ax.scatter(attn_stds, confidences, c=colors, alpha=0.6, s=30)
    ax.plot(x_line, m * x_line + b, color="black", lw=1.5, ls="--")
    ax.set_xlabel("Attention std  (per patient)")
    ax.set_ylabel("|prob − 0.5|  (prediction confidence)")
    p_str_r = f"{p_val:.3f}" if p_val >= 0.001 else "<0.001"
    ax.set_title(f"Attn std vs Confidence\nr = {r_val:.3f},  p = {p_str_r}")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",    markersize=8, label="Recurrence"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=8, label="No-recurrence"),
    ], fontsize=8)

    ax = axes[1]
    ax.hist(attn_stds, bins=30, color="gray", alpha=0.7, edgecolor="white")
    ax.axvline(thresh_high, color="tomato",    ls="--", lw=1.5,
               label=f"High-std threshold ({thresh_high:.3f})")
    ax.axvline(thresh_low,  color="steelblue", ls="--", lw=1.5,
               label=f"Low-std  threshold ({thresh_low:.3f})")
    ax.set_xlabel("Attention std"); ax.set_ylabel("Count")
    ax.set_title("Distribution of per-patient attention std")
    ax.legend(fontsize=8)

    if title_suffix:
        fig.suptitle(f"Non-uniform Attention Analysis\n{title_suffix}", y=1.01)
    plt.tight_layout()
    path_a = os.path.join(out_dir, "nonuniform_attn_std_vs_confidence.png")
    plt.savefig(path_a, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [NUA][a] Attn std vs confidence → {path_a}")

    # ── Shared helpers ────────────────────────────────────────────────────────
    model.eval()
    top_cases   = sorted_recs[:n_top_cases]
    use_bg      = top_cases[0].get("patch_types") is not None
    line_colors = plt.cm.tab10(np.linspace(0, 1, len(top_cases)))

    def _logit(p):
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    def _remove_drops(recs, indices_per_rec, desc="removal"):
        """Zero out specified patches per patient; return log-odds drops."""
        drops = []
        for r, indices in tqdm(zip(recs, indices_per_rec),
                               total=len(recs), desc=f"    {desc}", leave=False):
            ct_m = r["ct"].clone()
            pm   = r["pet"].clone()  if r.get("pet")  is not None else None
            dm   = r["dose"].clone() if r.get("dose") is not None else None
            for idx in indices:
                ct_m[idx] = 0.
                if pm is not None: pm[idx] = 0.
                if dm is not None: dm[idx] = 0.
            with torch.no_grad():
                probs_mod = _batched_infer(model, device, [ct_m],
                                           [pm] if pm is not None else [None],
                                           [dm] if dm is not None else [None],
                                           infer_batch_size, use_amp)
            drops.append(_logit(r["prob"]) - _logit(probs_mod[0]))
        return drops

    def _patient_sensitivity(recs):
        """Mean |single-patch drop| across all patches, per patient.

        Used to normalise removal drops so that baseline prediction confidence
        and per-patient model sensitivity do not confound comparisons.
        Returns a 1-D array of length len(recs); values are clipped to ≥1e-6.
        """
        sens = []
        for r in tqdm(recs, desc="    patient-sensitivity", leave=False):
            n_p = len(r["attn"])
            single = _remove_drops([r] * n_p, [[i] for i in range(n_p)],
                                   desc="")
            sens.append(max(float(np.mean(np.abs(single))), 1e-6))
        return np.array(sens)

    def _offset_label(patch_idx, patch_types=None):
        """Human-readable direction label for a patch index."""
        if patch_types is not None:
            return f"S{patch_idx // 2}-{'T' if patch_idx % 2 == 0 else 'B'}"
        if patch_idx >= len(MIL_OFFSETS):
            return f"P{patch_idx}"
        dx, dy, dz = MIL_OFFSETS[patch_idx]
        if dx == 0 and dy == 0 and dz == 0:
            return "center"
        dx_mm = dx * SPACING[0]; dy_mm = dy * SPACING[1]; dz_mm = dz * SPACING[2]
        _, axis, mag = max((abs(dx_mm), "X", dx_mm),
                           (abs(dy_mm), "Y", dy_mm),
                           (abs(dz_mm), "Z", dz_mm))
        return f"{'+' if mag > 0 else '-'}{axis} {abs(mag):.0f}mm"

    # ── (b) Top-k vs Bottom-k removal — paired comparison ────────────────────
    # Removing ANY patch drops the prediction slightly.  The key question is:
    # does removing the HIGH-attention patches hurt MORE than removing the
    # LOW-attention patches?  A paired Wilcoxon test controls for patient-level
    # baseline, isolating whether attention rank ↔ information contribution.
    K_VALUES  = [1, 2, 3]
    csv_rows_b = []
    top_drops_all = {}   # k → list of drops for each top_case
    bot_drops_all = {}

    # Per-patient sensitivity baseline (computed once, reused for all k)
    if normalize_by_patient_baseline:
        sens_top = _patient_sensitivity(top_cases)
    else:
        sens_top = np.ones(len(top_cases))

    for k in K_VALUES:
        top_idx_lists = [np.argsort(r["attn"])[::-1][:k] for r in top_cases]
        bot_idx_lists = [np.argsort(r["attn"])[:k]       for r in top_cases]
        top_drops_all[k] = _remove_drops(top_cases, top_idx_lists, f"top-{k}")
        bot_drops_all[k] = _remove_drops(top_cases, bot_idx_lists, f"bot-{k}")
        for r, td, bd, s in zip(top_cases, top_drops_all[k], bot_drops_all[k],
                                 sens_top):
            csv_rows_b.append({"pid": r["pid"], "k": k,
                                "top_k_drop": td, "bot_k_drop": bd,
                                "patient_sensitivity": s,
                                "top_k_drop_norm": td / s,
                                "bot_k_drop_norm": bd / s})

    y_label_b = ("Normalized log-odds drop\n(÷ patient avg single-patch drop)"
                 if normalize_by_patient_baseline else "Log-odds drop")

    fig, axes_b = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5))
    wilcoxon_results = {}
    for ax, k in zip(axes_b, K_VALUES):
        td_raw = top_drops_all[k]; bd_raw = bot_drops_all[k]
        td = [d / s for d, s in zip(td_raw, sens_top)]
        bd = [d / s for d, s in zip(bd_raw, sens_top)]
        # Gray connecting lines (paired)
        for t, b in zip(td, bd):
            ax.plot([0, 1], [t, b], color="gray", alpha=0.4, lw=1)
        ax.scatter([0] * len(td), td, color="tomato",    s=60, zorder=5,
                   label=f"top-{k} (high attn)")
        ax.scatter([1] * len(bd), bd, color="steelblue", s=60, zorder=5,
                   label=f"bottom-{k} (low attn)")
        ax.errorbar([0, 1], [np.mean(td), np.mean(bd)],
                    yerr=[np.std(td), np.std(bd)],
                    fmt="D", color="black", ms=8, capsize=5, lw=2, zorder=6)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"Top-{k}\n(high attn)", f"Bottom-{k}\n(low attn)"])
        ax.set_ylabel(y_label_b)
        if len(td) >= 2:
            stat_w, p_w = scipy_stats.wilcoxon(td, bd, alternative="greater")
            p_str_w = f"{p_w:.3f}" if p_w >= 0.001 else "<0.001"
            wilcoxon_results[k] = p_w
        else:
            p_str_w = "n/a"
        norm_tag = "  [norm]" if normalize_by_patient_baseline else ""
        title_k = f"k={k}   Wilcoxon p={p_str_w}{norm_tag}\n(top > bottom?)"
        if title_suffix: title_k += f"\n{title_suffix}"
        ax.set_title(title_k)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path_b = os.path.join(out_dir, "nonuniform_topk_removal_comparison.png")
    plt.savefig(path_b, dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(csv_rows_b).to_csv(
        os.path.join(out_dir, "nonuniform_topk_removal_comparison.csv"), index=False)
    print(f"  [NUA][b] Top-k vs Bottom-k removal → {path_b}")

    # ── (b_group) Top-k removal: top-N vs bottom-N std groups ───────────────
    # Removes top-k HIGH-attention patches from two groups of patients:
    # those with the highest attn_std (non-uniform) vs lowest attn_std (uniform).
    # Mann-Whitney tests whether high-std patients suffer more loss, proving
    # attention non-uniformity correlates with patch information contribution.
    # When normalize_by_patient_baseline=True each patient's drop is divided by
    # their mean |single-patch drop|, removing baseline-confidence confounds.
    n_grp      = min(n_std_group, len(sorted_recs) // 2)
    high_group = sorted_recs[:n_grp]
    low_group  = sorted_recs[-n_grp:]

    if normalize_by_patient_baseline:
        sens_hi = _patient_sensitivity(high_group)
        sens_lo = _patient_sensitivity(low_group)
    else:
        sens_hi = np.ones(n_grp)
        sens_lo = np.ones(n_grp)

    y_label_bg = ("Normalized log-odds drop\n(÷ patient avg single-patch drop)"
                  if normalize_by_patient_baseline else "Log-odds drop")

    fig, axes_bg = plt.subplots(1, len(K_VALUES), figsize=(5 * len(K_VALUES), 5))
    csv_rows_bg  = []
    mw_results   = {}

    for ax, k in zip(axes_bg, K_VALUES):
        hi_idx_lists = [np.argsort(r["attn"])[::-1][:k] for r in high_group]
        lo_idx_lists = [np.argsort(r["attn"])[::-1][:k] for r in low_group]
        hi_drops_raw = _remove_drops(high_group, hi_idx_lists, f"hi-grp top-{k}")
        lo_drops_raw = _remove_drops(low_group,  lo_idx_lists, f"lo-grp top-{k}")
        hi_drops = [d / s for d, s in zip(hi_drops_raw, sens_hi)]
        lo_drops = [d / s for d, s in zip(lo_drops_raw, sens_lo)]

        for r, d_raw, d_norm, s in zip(high_group, hi_drops_raw, hi_drops,
                                        sens_hi):
            csv_rows_bg.append({"pid": r["pid"], "group": "high_std", "k": k,
                                 "attn_std": r["attn_std"],
                                 "logodds_drop": d_raw,
                                 "patient_sensitivity": s,
                                 "logodds_drop_norm": d_norm})
        for r, d_raw, d_norm, s in zip(low_group, lo_drops_raw, lo_drops,
                                        sens_lo):
            csv_rows_bg.append({"pid": r["pid"], "group": "low_std", "k": k,
                                 "attn_std": r["attn_std"],
                                 "logodds_drop": d_raw,
                                 "patient_sensitivity": s,
                                 "logodds_drop_norm": d_norm})

        stat_mw, p_mw = scipy_stats.mannwhitneyu(
            hi_drops, lo_drops, alternative="greater")
        p_str_mw = f"{p_mw:.3f}" if p_mw >= 0.001 else "<0.001"
        mw_results[k] = p_mw

        bp = ax.boxplot(
            [hi_drops, lo_drops],
            tick_labels=[f"High-std\n(n={n_grp})", f"Low-std\n(n={n_grp})"],
            patch_artist=True, widths=0.5,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor("tomato");    bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_facecolor("steelblue"); bp["boxes"][1].set_alpha(0.7)
        for xpos, vals in enumerate([hi_drops, lo_drops], start=1):
            ax.scatter(
                [xpos + np.random.uniform(-0.08, 0.08) for _ in vals],
                vals, color="black", alpha=0.5, s=20, zorder=3,
            )
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_ylabel(y_label_bg)
        norm_tag = "  [norm]" if normalize_by_patient_baseline else ""
        title_bg = (f"Top-{k} removal{norm_tag}\n"
                    f"Mann-Whitney p={p_str_mw}  (high-std > low-std?)")
        if title_suffix: title_bg += f"\n{title_suffix}"
        ax.set_title(title_bg)

    plt.tight_layout()
    path_bg = os.path.join(out_dir, "nonuniform_std_group_removal_comparison.png")
    plt.savefig(path_bg, dpi=150, bbox_inches="tight"); plt.close()
    pd.DataFrame(csv_rows_bg).to_csv(
        os.path.join(out_dir, "nonuniform_std_group_removal_comparison.csv"), index=False)
    print(f"  [NUA][b_g] High-std(n={n_grp}) vs Low-std(n={n_grp}) "
          f"top-k removal → {path_bg}")

    # ── (b2) Ring analysis — attention by patch distance, top_cases only ─────
    if not use_bg:
        order, dist_mm, _ = _patch_order()
        N_p = len(order)
        attn_matrix = np.array([r["attn"][order] for r in top_cases])

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, r in enumerate(top_cases):
            ax.plot(range(N_p), attn_matrix[i], color=line_colors[i],
                    alpha=0.5, lw=1.2, marker="o", ms=5, label=r["pid"])
        ax.plot(range(N_p), attn_matrix.mean(axis=0), color="black",
                lw=2.5, marker="D", ms=8, label="mean", zorder=5)
        ax.axhline(1 / N_p, color="gray", ls="--", lw=1,
                   label=f"uniform = {1/N_p:.3f}")
        xlabels = [f"P{order[i]}\n{dist_mm[order[i]]:.0f}mm" for i in range(N_p)]
        ax.set_xticks(range(N_p)); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Attention weight")
        title_r = f"Attention by patch distance: {len(top_cases)} most non-uniform patients"
        if title_suffix: title_r += f"\n{title_suffix}"
        ax.set_title(title_r)
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        path_r = os.path.join(out_dir, "nonuniform_ring_analysis.png")
        plt.savefig(path_r, dpi=150, bbox_inches="tight"); plt.close()
        ring_rows = [
            {"pid": r["pid"], "patch_rank": rank, "patch_idx": int(order[rank]),
             "dist_mm": float(dist_mm[order[rank]]),
             "attn": float(r["attn"][order[rank]])}
            for r in top_cases for rank in range(N_p)
        ]
        pd.DataFrame(ring_rows).to_csv(
            os.path.join(out_dir, "nonuniform_ring_analysis.csv"), index=False)
        print(f"  [NUA][b2] Ring analysis (top-{n_top_cases} cases) → {path_r}")
    else:
        print(f"  [NUA][b2] BG-patch mode: ring analysis skipped.")

    # ── (c) Per-case PNG: top-3 patches (CT + PET rows, rank columns) ─────────
    N_RANKS  = 3
    cell     = 4.0
    vis_dir  = os.path.join(out_dir, "nonuniform_cases")
    os.makedirs(vis_dir, exist_ok=True)
    offset_csv_rows = []

    for r in top_cases:
        has_pet_r   = r.get("pet") is not None
        label_str   = "R" if r["label"] == 1 else "NR"
        top3_indices = np.argsort(r["attn"])[::-1][:N_RANKS]
        n_rows_vis   = 2 if has_pet_r else 1

        fig, axes_p = plt.subplots(
            n_rows_vis, N_RANKS,
            figsize=(cell * N_RANKS, cell * n_rows_vis),
            squeeze=False,
        )
        fig.subplots_adjust(left=0.01, right=0.99,
                            top=0.84, bottom=0.01,
                            wspace=0.04, hspace=0.04)

        for col, patch_idx in enumerate(top3_indices):
            offset_lbl = _offset_label(int(patch_idx), r.get("patch_types"))
            attn_val   = float(r["attn"][patch_idx])
            mid_z      = r["ct"].shape[2] // 2   # (N,1,D,H,W) → D dim

            ct_sl = r["ct"][patch_idx, 0, mid_z].numpy()
            axes_p[0, col].imshow(ct_sl, cmap="gray",
                                  vmin=ct_sl.min(), vmax=ct_sl.max(),
                                  aspect="equal", interpolation="nearest")
            axes_p[0, col].axis("off")
            axes_p[0, col].set_title(
                f"#{col+1} [{offset_lbl}]\nattn={attn_val:.3f}", fontsize=9)
            if col == 0:
                axes_p[0, col].set_ylabel("CT", fontsize=9)

            if has_pet_r:
                pet_sl = r["pet"][patch_idx, 0, mid_z].numpy()
                axes_p[1, col].imshow(pet_sl, cmap="hot",
                                      vmin=pet_sl.min(), vmax=pet_sl.max(),
                                      aspect="equal", interpolation="nearest")
                axes_p[1, col].axis("off")
                if col == 0:
                    axes_p[1, col].set_ylabel("PET", fontsize=9)

            offset_csv_rows.append({
                "pid":              r["pid"],
                "label":            r["label"],
                "prob":             r["prob"],
                "attn_std":         r["attn_std"],
                "rank":             col + 1,
                "patch_idx":        int(patch_idx),
                "offset_direction": offset_lbl,
                "attn_weight":      attn_val,
            })

        fig.suptitle(
            f"{r['pid']}   attn_std={r['attn_std']:.3f}   "
            f"prob={r['prob']:.2f} ({label_str})"
            + (f"   {title_suffix}" if title_suffix else ""),
            fontsize=9, y=0.97,
        )
        safe_pid = r["pid"].replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(vis_dir, f"case_{safe_pid}.png"), dpi=150)
        plt.close()

    pd.DataFrame(offset_csv_rows).to_csv(
        os.path.join(out_dir, "nonuniform_top3_offsets.csv"), index=False)
    print(f"  [NUA][c] Per-case PNGs ({len(top_cases)} files) → {vis_dir}/")
    print(f"  [NUA][c] Top-3 offset CSV → "
          f"{os.path.join(out_dir, 'nonuniform_top3_offsets.csv')}")

    # ── CSV (top_cases summary) ────────────────────────────────────────────────
    rows = []
    for i, r in enumerate(top_cases):
        rows.append({"pid": r["pid"], "label": r["label"], "prob": r["prob"],
                     "attn_std": r["attn_std"], "confidence": r["confidence"],
                     "top1_logodds_drop": top_drops_all[1][i],
                     "bot1_logodds_drop": bot_drops_all[1][i]})
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "nonuniform_analysis.csv"), index=False)
    print(f"  [NUA]    Summary CSV → {os.path.join(out_dir, 'nonuniform_analysis.csv')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  [NUA] Results:")
    print(f"         Pearson r(std, confidence) = {r_val:.3f},  p = {p_str_r}")
    for k in K_VALUES:
        td = top_drops_all[k]; bd = bot_drops_all[k]
        p_w = wilcoxon_results.get(k, float("nan"))
        p_str = f"{p_w:.3f}" if not np.isnan(p_w) and p_w >= 0.001 else (
                "<0.001" if not np.isnan(p_w) else "n/a")
        print(f"         k={k}: top-k drop={np.mean(td):.3f}  "
              f"bot-k drop={np.mean(bd):.3f}  Wilcoxon p={p_str}")

    return {
        "pearson_r":  float(r_val),
        "pearson_p":  float(p_val),
        "n_top_cases": len(top_cases),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Mode runners
# ══════════════════════════════════════════════════════════════════════════════

def _analyse_one(model, dataset, device, out_dir, args, title_suffix=""):
    """Run analyses for a single (model, device) pair.

    When args.nonuniform_only is True only export_attention and
    nonuniform_attention_analysis are executed; ring / topk / modality
    analyses are skipped and their return slots are filled with None.
    """
    use_bg        = getattr(args, "use_bg_patches", False)
    nonuniform_only = getattr(args, "nonuniform_only", False)
    os.makedirs(out_dir, exist_ok=True)
    records = export_attention(
        model, dataset, device, out_dir,
        batch_size=args.export_batch_size,
        num_workers=args.num_workers,
        use_amp=args.amp,
        use_bg_patches=use_bg,
    )

    if nonuniform_only:
        ring_stats, topk_stats, k_values, mod_stats = None, None, None, None
    else:
        if use_bg:
            ring_stats = tumor_vs_bg_analysis(records, out_dir, title_suffix)
        else:
            ring_stats = ring_analysis(records, out_dir, title_suffix)
        topk_stats, k_values = topk_removal(
            model, records, device, out_dir,
            n_rand_trials=args.n_rand_trials,
            infer_batch_size=args.infer_batch_size,
            use_amp=args.amp,
            title_suffix=title_suffix,
        )
        mod_stats = modality_ablation(
            model, records, device, out_dir,
            infer_batch_size=args.infer_batch_size,
            use_amp=args.amp,
            title_suffix=title_suffix,
        )

    nonuniform_attention_analysis(
        model, records, device, out_dir,
        infer_batch_size=args.infer_batch_size,
        use_amp=args.amp,
        n_top_cases=getattr(args, "n_top_cases", 6),
        n_std_group=getattr(args, "n_std_group", 10),
        title_suffix=title_suffix,
        normalize_by_patient_baseline=not getattr(
            args, "no_patient_baseline_norm", False),
    )
    return records, ring_stats, topk_stats, k_values, mod_stats


def run_single(args, model, dataset, device):
    use_bg    = getattr(args, "use_bg_patches", False)
    n_patches = (2 * getattr(args, "n_slices", 5)) if use_bg else len(MIL_OFFSETS)
    print(f"\n{'='*60}")
    print(f"  MIL mechanistic analysis  [single-model mode]")
    print(f"  Patients    : {len(dataset)}")
    print(f"  Patches/pt  : {n_patches}"
          + ("  (tumor+bg slice bags)" if use_bg else "  (offset bags)"))
    print(f"  BG patches  : {use_bg}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  AMP         : {args.amp}")
    print(f"  Output dir  : {args.out_dir}")
    print(f"{'='*60}\n")
    _analyse_one(model, dataset, device, args.out_dir, args)
    print(f"\nDone. All outputs in {args.out_dir}/")


# ── Multi-GPU worker (called in a subprocess) ─────────────────────────────────

def _worker(rank, ckpt, dataset, gpu_id, sub_dir, args_dict, result_queue):
    """
    Runs on a single process assigned to gpu_id.
    Puts (ring_stats, topk_stats, k_values, mod_stats, records) onto result_queue.
    """
    import types
    args = types.SimpleNamespace(**args_dict)
    # 子进程内不能再 fork DataLoader workers（会成为守护进程的孙进程），强制单线程加载
    args.num_workers = 0

    device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() \
             else torch.device("cpu")

    if args.model_size == "small":
        _dla, _fus, _dose_ch = DLA_CHANNELS_SMALL, FUSION_HIDDEN_SMALL, DOSE_CHANNELS_SMALL
    else:
        _dla, _fus, _dose_ch = DLA_CHANNELS, FUSION_HIDDEN, DOSE_CHANNELS

    model = load_model(
        ckpt, device,
        use_dose=args.use_dose, use_mil=True,
        dla_channels=_dla, fusion_hidden=_fus, dose_channels=_dose_ch,
    )
    suffix = os.path.basename(ckpt)
    records, ring_stats, topk_stats, k_values, mod_stats = \
        _analyse_one(model, dataset, device, sub_dir, args, title_suffix=suffix)

    # Strip tensors from records before sending through queue (not picklable on all platforms)
    slim_records = [{"pid": r["pid"], "label": r["label"],
                     "prob": r["prob"], "attn": r["attn"],
                     "patch_types": r.get("patch_types")} for r in records]
    result_queue.put((rank, slim_records, ring_stats, topk_stats, k_values, mod_stats))


def run_multi(args, checkpoints, dataset, device_single, _dla, _fus, _dose_ch):
    """
    dataset: 单个 PatchDataset（所有模型共用）或 list[PatchDataset]（每模型用各自的 val 集）。
    """
    out_dir  = args.out_dir
    n_models = len(checkpoints)
    n_gpus   = torch.cuda.device_count()
    os.makedirs(out_dir, exist_ok=True)

    # 统一为列表，方便按索引取
    if isinstance(dataset, list):
        ds_list = dataset
        assert len(ds_list) == n_models, "dataset list length must match checkpoints"
    else:
        ds_list = [dataset] * n_models

    print(f"\n{'='*60}")
    print(f"  MIL mechanistic analysis  [multi-model mode: {n_models} checkpoints]")
    pt_counts = "/".join(str(len(ds)) for ds in ds_list)
    print(f"  Patients    : {pt_counts}  (per model)")
    use_bg = getattr(args, "use_bg_patches", False)
    n_patches_display = (2 * getattr(args, "n_slices", 5)) if use_bg else len(MIL_OFFSETS)
    print(f"  Patches/pt  : {n_patches_display}")
    print(f"  GPUs avail  : {n_gpus}  {'→ models distributed across GPUs' if n_gpus > 1 else '(single GPU)'}")
    print(f"  AMP         : {args.amp}")
    print(f"  Val only    : {getattr(args, 'val_only', False)}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'='*60}\n")

    sub_dirs = [os.path.join(out_dir, f"model_{i}") for i in range(n_models)]

    if n_gpus > 1:
        # ── Parallel multi-GPU: each model on its own GPU ──────────────────
        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()
        args_dict = vars(args)

        procs = []
        for idx, (ckpt, sub_dir) in enumerate(zip(checkpoints, sub_dirs)):
            gpu_id = idx % n_gpus
            print(f"  Spawning model_{idx} → GPU {gpu_id}  ({ckpt})")
            p = mp.Process(
                target=_worker,
                args=(idx, ckpt, ds_list[idx], gpu_id, sub_dir, args_dict, result_queue),
                daemon=False,  # daemon=True 禁止子进程创建孙进程(DataLoader workers)
            )
            p.start(); procs.append(p)

        # Collect results in completion order, then sort by rank
        raw_results = [result_queue.get() for _ in range(n_models)]
        for p in procs:
            p.join()
        raw_results.sort(key=lambda x: x[0])
        all_records  = [r[1] for r in raw_results]
        all_ring     = [r[2] for r in raw_results]
        all_topk     = [r[3] for r in raw_results]
        k_values     = raw_results[0][4]
        all_mod      = [r[5] for r in raw_results]

    else:
        # ── Sequential single-GPU (or CPU) ─────────────────────────────────
        all_records, all_ring, all_topk, all_mod = [], [], [], []
        k_values = None

        for idx, (ckpt, sub_dir) in enumerate(zip(checkpoints, sub_dirs)):
            print(f"\n── Model {idx + 1}/{n_models}: {ckpt}")
            model = load_model(
                ckpt, device_single,
                use_dose=args.use_dose, use_mil=True,
                dla_channels=_dla, fusion_hidden=_fus, dose_channels=_dose_ch,
            )
            suffix = os.path.basename(ckpt)
            records, ring_stats, topk_stats, kv, mod_stats = \
                _analyse_one(model, ds_list[idx], device_single, sub_dir, args,
                             title_suffix=suffix)
            slim_records = [{"pid": r["pid"], "label": r["label"],
                             "prob": r["prob"], "attn": r["attn"]} for r in records]
            all_records.append(slim_records)
            all_ring.append(ring_stats)
            all_topk.append(topk_stats)
            k_values = kv
            all_mod.append(mod_stats)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Cross-model averages ───────────────────────────────────────────────────
    avg_dir = os.path.join(out_dir, "average")
    os.makedirs(avg_dir, exist_ok=True)
    print(f"\n── Computing averages across {n_models} models → {avg_dir}/")
    use_bg          = getattr(args, "use_bg_patches", False)
    nonuniform_only = getattr(args, "nonuniform_only", False)
    _avg_attention(all_records, avg_dir)
    if nonuniform_only:
        print("[avg] nonuniform_only mode: ring / topk / modality averages skipped.")
    else:
        if use_bg:
            print("[avg][2] BG-patch mode: per-model tumor-vs-bg plots already saved; "
                  "cross-model ring average skipped.")
        else:
            _avg_ring(all_ring, avg_dir, n_models)
        _avg_topk(all_topk, k_values, avg_dir, n_models)
        _avg_modality(all_mod, avg_dir, n_models)

    print(f"\nDone.")
    print(f"  Per-model outputs : {out_dir}/model_{{0..{n_models-1}}}/")
    print(f"  Averaged outputs  : {avg_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mechanistic analysis of MIL attention weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Mode switch ────────────────────────────────────────────────────────────
    ck = p.add_mutually_exclusive_group(required=True)
    ck.add_argument("--checkpoint",  metavar="PATH",
                    help="Single checkpoint → single-model mode.")
    ck.add_argument("--checkpoints", metavar="PATH", nargs="+",
                    help="Multiple checkpoints → multi-model mode (+ averaged output).")
    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--csv",             required=True)
    p.add_argument("--ct_dir",          required=True)
    p.add_argument("--pet_dir",         default=None)
    p.add_argument("--mask_dir",        default=None)
    p.add_argument("--dose_dir",        default=None)
    p.add_argument("--label_col",       default="recurrence",
                   choices=["recurrence", "figo"])
    p.add_argument("--ct_axis_order",   default="ZYX", choices=["ZYX", "XYZ"])
    p.add_argument("--pet_axis_order",  default="XYZ", choices=["ZYX", "XYZ"])
    p.add_argument("--dose_axis_order", default="ZYX", choices=["ZYX", "XYZ"])
    p.add_argument("--ct_wl",    type=float, default=40.0)
    p.add_argument("--ct_ww",    type=float, default=400.0)
    p.add_argument("--pet_max",  type=float, default=200.0)
    p.add_argument("--dose_max", type=float, default=1.0)
    p.add_argument("--use_dose",       action="store_true")
    p.add_argument("--use_bg_patches", action="store_true",
                   help="Analyse a model trained with --use_bg_patches "
                        "(slice-level tumor+bg bags). Enables tumor-vs-bg "
                        "attention analysis and disables ring analysis.")
    p.add_argument("--n_slices",       type=int,   default=5,
                   help="K slices per patient in BG-patch bags "
                        "(must match training, default 5).")
    p.add_argument("--bg_min_dist_mm", type=float, default=50.0,
                   help="Min distance from tumour mask for BG patch centres "
                        "(must match training, default 50 mm).")
    p.add_argument("--model_size", default="base", choices=["base", "small"])
    p.add_argument("--out_dir",    default="outputs/mil_analysis")
    p.add_argument("--seed",       type=int, default=42)
    # ── Val-only split ────────────────────────────────────────────────────────
    p.add_argument("--val_only", action="store_true",
                   help="Run analysis only on each fold's validation patients "
                        "(reproduces the StratifiedKFold split used in training).")
    p.add_argument("--fold", type=int, default=None,
                   help="Fold index (1-based) for --val_only in single-model mode.")
    p.add_argument("--num_folds", type=int, default=5,
                   help="Total number of folds (must match training, default 5).")
    # ── Speed ─────────────────────────────────────────────────────────────────
    p.add_argument("--amp", action="store_true",
                   help="Enable float16 AMP inference (~2× GPU throughput).")
    p.add_argument("--infer_batch_size", type=int, default=128,
                   help="Max variants per GPU call in topk/modality (default 128). "
                        "Reduce if OOM.")
    p.add_argument("--export_batch_size", type=int, default=4,
                   help="Patient batch size for export_attention (default 4).")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers for export_attention (default 4).")
    p.add_argument("--n_rand_trials", type=int, default=10,
                   help="Random-k trials per k per patient in topk_removal (default 10).")
    p.add_argument("--n_top_cases", type=int, default=6,
                   help="Number of most non-uniform cases to visualize (default 6).")
    p.add_argument("--n_std_group", type=int, default=10,
                   help="Patients per group in high-std vs low-std removal comparison (default 10).")
    p.add_argument("--no_patient_baseline_norm", action="store_true",
                   help="Disable per-patient sensitivity normalisation in nonuniform "
                        "removal comparisons (raw log-odds drops instead of normalised).")
    p.add_argument("--nonuniform_only", action="store_true",
                   help="Skip ring / topk / modality analyses; run only "
                        "nonuniform_attention_analysis (faster).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pet_dir = args.pet_dir or args.ct_dir

    if args.model_size == "small":
        _dla, _fus, _dose_ch = DLA_CHANNELS_SMALL, FUSION_HIDDEN_SMALL, DOSE_CHANNELS_SMALL
    else:
        _dla, _fus, _dose_ch = DLA_CHANNELS, FUSION_HIDDEN, DOSE_CHANNELS

    def _make_dataset(patient_ids=None):
        return PatchDataset(
            csv_path        = args.csv,
            ct_dir          = args.ct_dir,
            pet_dir         = pet_dir,
            mask_dir        = args.mask_dir,
            label_col       = args.label_col,
            ct_axis_order   = args.ct_axis_order,
            pet_axis_order  = args.pet_axis_order,
            ct_wl           = args.ct_wl,
            ct_ww           = args.ct_ww,
            pet_max         = args.pet_max,
            use_dose        = args.use_dose,
            dose_dir        = args.dose_dir,
            dose_axis_order = args.dose_axis_order,
            dose_max        = args.dose_max,
            use_mil         = True,
            mil_patch_size  = MIL_PATCH_SIZE,
            mil_offsets     = MIL_OFFSETS,
            augment         = False,
            patient_ids     = patient_ids,
            use_bg_patches  = args.use_bg_patches,
            n_slices        = args.n_slices,
            bg_min_dist_mm  = args.bg_min_dist_mm,
        )

    if args.val_only:
        if args.checkpoint:
            # 单模型：必须指定 --fold
            if args.fold is None:
                raise ValueError("--val_only in single-model mode requires --fold (e.g. --fold 1)")
            val_ids = get_val_ids(args.csv, args.label_col, args.fold,
                                  args.num_folds, args.seed)
            print(f"[val_only] fold {args.fold}: {len(val_ids)} val patients")
            dataset = _make_dataset(patient_ids=val_ids)
        else:
            # 多模型：fold i+1 对应 checkpoints[i]
            n_ckpts = len(args.checkpoints)
            dataset = []
            for fi in range(1, n_ckpts + 1):
                val_ids = get_val_ids(args.csv, args.label_col, fi,
                                      args.num_folds, args.seed)
                print(f"[val_only] fold {fi}: {len(val_ids)} val patients")
                dataset.append(_make_dataset(patient_ids=val_ids))
    else:
        dataset = _make_dataset()

    if args.checkpoint:
        model = load_model(
            args.checkpoint, device,
            use_dose=args.use_dose, use_mil=True,
            dla_channels=_dla, fusion_hidden=_fus, dose_channels=_dose_ch,
        )
        run_single(args, model, dataset, device)
    else:
        run_multi(args, args.checkpoints, dataset, device, _dla, _fus, _dose_ch)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--checkpoint",     "outputs/fold1_best.pth",
            "--csv",            "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv",
            "--ct_dir",         "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy",
            "--pet_dir",        "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy",
            "--mask_dir",       "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled",
            "--ct_axis_order",  "ZYX",
            "--pet_axis_order", "XYZ",
            "--ct_wl",          "40",
            "--ct_ww",          "400",
            "--pet_max",        "200",
            "--out_dir",        "outputs/mil_analysis",
            "--amp",
        ]
    main()
