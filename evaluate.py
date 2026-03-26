"""
Evaluation utilities:
  - Run inference on a test set with a saved model checkpoint.
  - Plot ROC curve.
  - Print / save a full metrics table.

Usage:
    python evaluate.py \
        --checkpoint outputs/fold0_best.pth \
        --data_root /data \
        --label_csv /data/labels.csv \
        --patient_ids p001 p002 ...
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc as sk_auc
from tqdm import tqdm

from config import (
    BATCH_SIZE, DLA_CHANNELS, FUSION_HIDDEN, DROPOUT, OUTPUT_DIR,
    USE_DOSE, DOSE_CHANNELS, USE_MIL, MIL_ATTN_DIM,
    CT_LARGE_CHANNELS, DOSE_LARGE_CHANNELS, DUAL_SCALE_FUSION_HIDDEN,
    DUAL_SCALE_USE_CT_LARGE,
    MISSING_GATE_HIDDEN,
)
from model import DualBranch3DCNN, DualScaleModel
from dataset import PatchDataset
from utils import compute_metrics, find_optimal_threshold


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_dual_scale: bool = False,
    use_ct_only: bool = False,
    use_pet_only: bool = False,
    use_missing_gate: bool = False,
    use_global_branch: bool = False,
    zero_ct: bool = False,
    zero_pet: bool = False,
    use_unipair: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return (y_true, y_prob, pet_present_arr).
    pet_present_arr is None unless use_missing_gate=True or use_unipair=True
    (in the unipair case it holds the modality values: 0=CT, 1=PET).
    zero_ct=True  zeroes CT  before forward pass (PET-only ablation).
    zero_pet=True zeroes PET before forward pass (CT-only ablation on paired set)."""
    model.eval()
    all_probs, all_labels = [], []
    all_pet_present = [] if (use_missing_gate or use_unipair) else None
    if use_unipair:
        for batch in tqdm(loader, desc="Inference"):
            ct         = batch[0].to(device)
            pet        = batch[1].to(device)
            dose       = batch[2].to(device)
            labels     = batch[3]
            modality_b = batch[4].to(device)
            logits = model(ct, pet, dose, modality=modality_b)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_pet_present.extend(batch[4].numpy().tolist())
        pet_arr = np.array(all_pet_present)
        return np.array(all_labels), np.array(all_probs), pet_arr
    if use_dual_scale:
        for ct_small, pet_small, ct_large, dose_large, labels in tqdm(
                loader, desc="Inference"):
            logits = model(
                ct_small.to(device), pet_small.to(device),
                ct_large.to(device), dose_large.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    elif use_ct_only:
        for ct, labels in tqdm(loader, desc="Inference"):
            logits = model(ct.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    elif use_pet_only:
        for pet, labels in tqdm(loader, desc="Inference"):
            logits = model(pet.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    else:
        for batch in tqdm(loader, desc="Inference"):
            _b = list(batch)
            pet_present_b = _b.pop().to(device) if use_missing_gate else None
            if use_missing_gate:
                all_pet_present.extend(batch[-1].numpy().tolist())
            if use_global_branch:
                dose_global_b = _b.pop().to(device)
                ct_global_b   = _b.pop().to(device)
            else:
                ct_global_b = dose_global_b = None

            ct     = _b[0].to(device)
            if zero_ct:
                ct = torch.zeros_like(ct)
            pet    = _b[1].to(device)
            if zero_pet:
                pet = torch.zeros_like(pet)
                if use_missing_gate:
                    pet_present_b = torch.zeros(pet.shape[0], device=device)
            dose   = _b[2].to(device)
            labels = _b[3]

            if use_global_branch:
                logits = model(ct, pet, ct_global_b, dose_global_b,
                               pet_present=pet_present_b)
            else:
                logits = model(ct, pet, dose, pet_present=pet_present_b)

            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    pet_arr = np.array(all_pet_present) if all_pet_present is not None else None
    return np.array(all_labels), np.array(all_probs), pet_arr


def load_model(checkpoint_path: str, device: torch.device,
               use_dose:                 bool  = USE_DOSE,
               use_mil:                  bool  = USE_MIL,
               dla_channels:             list  = None,
               fusion_hidden:            int   = None,
               dose_channels:            list  = None,
               use_dual_scale:           bool  = False,
               ct_large_channels:        list  = None,
               dose_large_channels:      list  = None,
               dual_scale_fusion_hidden: int   = None,
               large_branch_use_ct:      bool  = False,
               use_ct_only:              bool  = False,
               use_pet_only:             bool  = False,
               use_missing_gate:         bool  = False,
               gate_hidden:              int   = None,
               use_unipair:              bool  = False):
    _dla_channels  = dla_channels  if dla_channels  is not None else DLA_CHANNELS
    _fusion_hidden = fusion_hidden if fusion_hidden is not None else FUSION_HIDDEN
    _dose_channels = dose_channels if dose_channels is not None else DOSE_CHANNELS
    _gate_hidden   = gate_hidden   if gate_hidden   is not None else MISSING_GATE_HIDDEN

    if use_dual_scale:
        _ct_large        = ct_large_channels        if ct_large_channels        is not None else CT_LARGE_CHANNELS
        _dose_large      = dose_large_channels      if dose_large_channels      is not None else DOSE_LARGE_CHANNELS
        _ds_fusion_hidden = dual_scale_fusion_hidden if dual_scale_fusion_hidden is not None else DUAL_SCALE_FUSION_HIDDEN
        model = DualScaleModel(
            dla_channels=_dla_channels,
            fusion_hidden=_fusion_hidden,
            dropout=DROPOUT,
            use_mil=use_mil,
            mil_attn_dim=MIL_ATTN_DIM,
            ct_large_channels=_ct_large,
            dose_large_channels=_dose_large,
            dual_scale_fusion_hidden=_ds_fusion_hidden,
            large_branch_use_ct=large_branch_use_ct,
        ).to(device)
    else:
        model = DualBranch3DCNN(
            dla_channels=_dla_channels,
            fusion_hidden=_fusion_hidden,
            dropout=DROPOUT,
            use_dose=use_dose,
            dose_channels=_dose_channels,
            use_mil=use_mil,
            mil_attn_dim=MIL_ATTN_DIM,
            ct_only=use_ct_only,
            pet_only=use_pet_only,
            use_missing_gate=use_missing_gate,
            gate_hidden=_gate_hidden,
            use_unipair=use_unipair,
        ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


# ── ROC plot ──────────────────────────────────────────────────────────────────

def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    save_path: str,
    title: str = "ROC Curve",
):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)

    # Find the operating point
    idx = np.argmin(np.abs(thresholds - threshold))

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"AUC = {roc_auc:.3f}")
    plt.scatter(fpr[idx], tpr[idx], color="red", zorder=5,
                label=f"threshold = {threshold:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("1 – Specificity (FPR)")
    plt.ylabel("Sensitivity (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC saved → {save_path}")


# ── Aggregate fold results ────────────────────────────────────────────────────

def aggregate_fold_results(fold_results: list[dict]) -> dict:
    """Average metrics across folds."""
    keys = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
    summary = {}
    for k in keys:
        vals = [r["val_metrics"][k] for r in fold_results]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))
    return summary


def print_summary(fold_results: list[dict], out_dir: str = OUTPUT_DIR):
    summary = aggregate_fold_results(fold_results)
    print("\n" + "=" * 60)
    print("Cross-validation summary  [Phase 1]")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<22s}: {v:.4f}")
    print("=" * 60)

    # ── Fine-tune summary (when available) ────────────────────────────────────
    ft_results = [r for r in fold_results if r.get("ft_val_metrics")]
    if ft_results:
        keys = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
        print("\n" + "=" * 60)
        print("Cross-validation summary  [Fine-tuned — paired val]")
        print("=" * 60)
        for k in keys:
            vals = [r["ft_val_metrics"][k] for r in ft_results
                    if r["ft_val_metrics"].get(k) is not None]
            if vals:
                print(f"  {k+'_mean':<22s}: {float(np.mean(vals)):.4f}  "
                      f"std={float(np.std(vals)):.4f}")
        print("=" * 60)

        ct_results = [r for r in ft_results if r.get("ft_val_metrics_ctonly")]
        if ct_results:
            print("\n" + "=" * 60)
            print("Cross-validation summary  [Fine-tuned — CT-only val]")
            print("=" * 60)
            for k in keys:
                vals = [r["ft_val_metrics_ctonly"][k] for r in ct_results
                        if r["ft_val_metrics_ctonly"].get(k) is not None]
                if vals:
                    print(f"  {k+'_mean':<22s}: {float(np.mean(vals)):.4f}  "
                          f"std={float(np.std(vals)):.4f}")
            print("=" * 60)

    # ── Drop-PET ablation summary (when available) ────────────────────────────
    dp_results = [r for r in fold_results if r.get("val_metrics_drop_pet")]
    if dp_results:
        keys = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
        print("\n" + "=" * 60)
        print("Cross-validation summary  [Phase 1 — CT-only ablation (PET zeroed)]")
        print("=" * 60)
        for k in keys:
            vals = [r["val_metrics_drop_pet"][k] for r in dp_results
                    if r["val_metrics_drop_pet"].get(k) is not None]
            if vals:
                print(f"  {k+'_mean':<22s}: {float(np.mean(vals)):.4f}  "
                      f"std={float(np.std(vals)):.4f}")
        print("=" * 60)

    ft_dp_results = [r for r in fold_results if r.get("ft_val_metrics_drop_pet")]
    if ft_dp_results:
        keys = ["auc", "sensitivity", "specificity", "ppv", "npv", "accuracy"]
        print("\n" + "=" * 60)
        print("Cross-validation summary  [Fine-tuned — CT-only ablation (PET zeroed)]")
        print("=" * 60)
        for k in keys:
            vals = [r["ft_val_metrics_drop_pet"][k] for r in ft_dp_results
                    if r["ft_val_metrics_drop_pet"].get(k) is not None]
            if vals:
                print(f"  {k+'_mean':<22s}: {float(np.mean(vals)):.4f}  "
                      f"std={float(np.std(vals)):.4f}")
        print("=" * 60)

    # ── Per-fold CSV ───────────────────────────────────────────────────────────
    rows = []
    for r in fold_results:
        row = {"fold": r["fold"], "phase": "phase1", **r["val_metrics"]}
        rows.append(row)
        if r.get("val_metrics_drop_pet"):
            rows.append({"fold": r["fold"], "phase": "phase1_drop_pet",
                         **r["val_metrics_drop_pet"]})
        if r.get("ft_val_metrics"):
            rows.append({"fold": r["fold"], "phase": "finetune_paired",
                         **r["ft_val_metrics"]})
        if r.get("ft_val_metrics_ctonly"):
            rows.append({"fold": r["fold"], "phase": "finetune_ctonly",
                         **r["ft_val_metrics_ctonly"]})
        if r.get("ft_val_metrics_drop_pet"):
            rows.append({"fold": r["fold"], "phase": "finetune_drop_pet",
                         **r["ft_val_metrics_drop_pet"]})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "cv_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Per-fold results saved → {csv_path}")

    # ── Paired vs CT-only ablation summary CSV ────────────────────────────────
    STAT_KEYS = ["auc", "accuracy", "specificity", "sensitivity"]
    has_paired   = any(r.get("val_metrics") for r in fold_results)
    has_drop_pet = any(r.get("val_metrics_drop_pet") for r in fold_results)
    if has_paired and has_drop_pet:
        abl_rows = []
        for r in fold_results:
            m_p  = r.get("val_metrics", {})
            m_dp = r.get("val_metrics_drop_pet", {})
            row = {"fold": r["fold"]}
            for k in STAT_KEYS:
                row[f"paired_{k}"]   = m_p.get(k, float("nan"))
                row[f"drop_pet_{k}"] = m_dp.get(k, float("nan"))
            abl_rows.append(row)
        # mean and std rows
        for stat, fn in [("mean", np.mean), ("std", np.std)]:
            row = {"fold": stat}
            for k in STAT_KEYS:
                paired_vals   = [r[f"paired_{k}"]   for r in abl_rows
                                 if not np.isnan(r[f"paired_{k}"])]
                drop_pet_vals = [r[f"drop_pet_{k}"] for r in abl_rows
                                 if not np.isnan(r[f"drop_pet_{k}"])]
                row[f"paired_{k}"]   = float(fn(paired_vals))   if paired_vals   else float("nan")
                row[f"drop_pet_{k}"] = float(fn(drop_pet_vals)) if drop_pet_vals else float("nan")
            abl_rows.append(row)
        abl_df = pd.DataFrame(abl_rows)
        abl_path = os.path.join(out_dir, "ablation_paired_vs_drop_pet.csv")
        abl_df.to_csv(abl_path, index=False)
        print(f"Paired vs CT-only ablation summary → {abl_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--data_root",   required=True)
    parser.add_argument("--label_csv",   required=True)
    parser.add_argument("--threshold",   type=float, default=None,
                        help="Fixed threshold; if omitted, use Youden on test set.")
    parser.add_argument("--out_dir",     default=OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df  = pd.read_csv(args.label_csv)
    ids = df["patient_id"].tolist()
    lbs = df["label"].tolist()

    ds = PatchDataset(
        data_root=args.data_root,
        patient_ids=ids,
        labels=lbs,
        augment=False,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model    = load_model(args.checkpoint, device)
    y_true, y_prob, _ = run_inference(model, loader, device)

    threshold = args.threshold or find_optimal_threshold(y_true, y_prob)
    metrics   = compute_metrics(y_true, y_prob, threshold)

    print("\nTest-set metrics")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:<14s}: {v}")

    plot_roc(
        y_true, y_prob, threshold,
        save_path=os.path.join(args.out_dir, "roc_test.png"),
        title="Test ROC",
    )

    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.out_dir, "test_metrics.csv"), index=False
    )


if __name__ == "__main__":
    main()
