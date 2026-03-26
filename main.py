"""
Main entry point: k-fold cross-validation training pipeline.

File layout expected:
    <ct_dir>/{Patient_ID}_CT.npy
    <pet_dir>/{Patient_ID}_PET.npy

CSV columns:
    label_col='recurrence' → Patient_ID, Recurrence (Yes/No or 1/0)
    label_col='figo'       → Patient_ID, FIGO 2018 Stage

Run:
    python main.py \
        --csv       /path/to/all_patients_info.csv \
        --ct_dir    /path/to/npy \
        --pet_dir   /path/to/npy \
        --label_col recurrence \
        --out_dir   outputs/run01
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from sklearn.model_selection import StratifiedKFold

from config import (NUM_FOLDS, SEED, MIXUP_PROB, MIXUP_ALPHA, OUTPUT_DIR,
                    USE_DOSE, DOSE_MAX, USE_MIL,
                    DLA_CHANNELS, FUSION_HIDDEN, DOSE_CHANNELS,
                    DLA_CHANNELS_SMALL, FUSION_HIDDEN_SMALL, DOSE_CHANNELS_SMALL,
                    PATCH_SIZE, LARGE_PATCH_SIZE, MIL_PATCH_SIZE, MIL_OFFSETS,
                    CT_LARGE_CHANNELS, DOSE_LARGE_CHANNELS, DUAL_SCALE_FUSION_HIDDEN,
                    DUAL_SCALE_USE_CT_LARGE,
                    PET_DROPOUT_PROB,
                    GLOBAL_SIZE, GLOBAL_ENC_CHANNELS, GLOBAL_OUT_DIM)
from dataset import PatchDataset, parse_labels
from train import train_fold
from evaluate import plot_roc, print_summary, run_inference, load_model
from utils import set_seed, compute_metrics, find_optimal_threshold
from torch.utils.data import DataLoader
from config import BATCH_SIZE


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    """
    Initialise process group when launched via torchrun.
    Returns (rank, local_rank, world_size).
    If not in a distributed environment (plain `python main.py`),
    returns (0, 0, 1) so single-GPU code path is unchanged.
    """
    rank       = int(os.environ.get("RANK",       0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Dual-branch 3D CNN — cross-val training")

    # ── Data paths ────────────────────────────────────────────────────────────
    p.add_argument("--csv",        required=True,
                   help="Path to CSV (Patient_ID + label column)")
    p.add_argument("--ct_dir",     required=True,
                   help="Directory containing {Patient_ID}_CT.npy files")
    p.add_argument("--pet_dir",    default=None,
                   help="Directory containing {Patient_ID}_PET.npy files "
                        "(defaults to --ct_dir if omitted)")
    p.add_argument("--mask_dir",   default=None,
                   help="Directory containing {Patient_ID}_MTV_Cervix.npy masks "
                        "(case-insensitive). Omit to fall back to volume centre.")
    p.add_argument("--use_dose",       action="store_true", default=False,
                   help="Enable Dose as third modality (tri-branch model).")
    p.add_argument("--use_mil",        action="store_true", default=False,
                   help="Enable MIL-Attention: N instances per patient instead of 1.")
    p.add_argument("--use_dual_scale", action="store_true", default=False,
                   help="Enable dual-scale branch: adds a large-patch CT+Dose branch "
                        "(120x120x90 mm). Large patch size is read from config.")
    p.add_argument("--use_ct_only",    action="store_true", default=False,
                   help="CT-only ablation: train with CT branch only, no PET/Dose. "
                        "Compatible with --use_mil for MIL-Attention mode.")
    p.add_argument("--use_pet_only",   action="store_true", default=False,
                   help="PET-only ablation: train with PET branch only, no CT/Dose. "
                        "Compatible with --use_mil for MIL-Attention mode.")
    p.add_argument("--dose_dir",   default=None,
                   help="Directory containing {Patient_ID}_Dose.npy files.")
    p.add_argument("--dose_axis_order", default="ZYX", choices=["ZYX", "XYZ"],
                   help="Axis order of Dose .npy files (default: ZYX, same as CT).")
    p.add_argument("--dose_max",   type=float, default=DOSE_MAX,
                   help=f"Dose clip maximum in Gy (default: {DOSE_MAX}).")

    # ── Label ─────────────────────────────────────────────────────────────────
    p.add_argument("--label_col",  default="recurrence",
                   choices=["recurrence", "figo"],
                   help="Which outcome column to predict (default: recurrence)")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    p.add_argument("--ct_axis_order",  default="ZYX", choices=["ZYX", "XYZ"],
                   help="Axis order of CT .npy files: ZYX→transpose, XYZ→keep (default: ZYX)")
    p.add_argument("--pet_axis_order", default="XYZ", choices=["ZYX", "XYZ"],
                   help="Axis order of PET .npy files: ZYX→transpose, XYZ→keep (default: XYZ)")
    p.add_argument("--ct_wl",      type=float, default=40.0,
                   help="CT window level in HU (default: 40)")
    p.add_argument("--ct_ww",      type=float, default=400.0,
                   help="CT window width in HU (default: 400)")
    p.add_argument("--pet_max",    type=float, default=10.0,
                   help="PET SUV clip maximum for [0,1] normalisation (default: 10)")

    # ── Experiment ────────────────────────────────────────────────────────────
    p.add_argument("--num_folds",   type=int, default=NUM_FOLDS)
    p.add_argument("--folds",       type=int, nargs="+", default=None,
                   help="Only run these fold indices, e.g. --folds 4 5 "
                        "(1-based). Omit to run all folds.")
    p.add_argument("--seed",        type=int, default=SEED)
    p.add_argument("--out_dir",     default=OUTPUT_DIR)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--preload",     action="store_true",
                   help="Preload all volumes into RAM before training")
    p.add_argument("--model_size",  default="base", choices=["base", "small"],
                   help="'base' ~26M params (default); 'small' ~6M params")
    p.add_argument("--patch_size",     type=int, nargs=3, default=list(PATCH_SIZE),
                   metavar=("X", "Y", "Z"),
                   help="Single-instance patch size in voxels X Y Z "
                        f"(default: {list(PATCH_SIZE)})")
    p.add_argument("--mil_patch_size", type=int, nargs=3, default=list(MIL_PATCH_SIZE),
                   metavar=("X", "Y", "Z"),
                   help="MIL instance patch size in voxels X Y Z "
                        f"(default: {list(MIL_PATCH_SIZE)})")

    # ── Missing modality gate ─────────────────────────────────────────────────
    p.add_argument("--use_missing_gate",   action="store_true", default=False,
                   help="Enable missing-aware PET gate. Allows CT-only patients "
                        "(no PET file) to be trained in the same loop as paired "
                        "CT+PET. Combined with --pet_dropout_prob for robustness.")
    p.add_argument("--pet_dropout_prob",   type=float, default=PET_DROPOUT_PROB,
                   help=f"Prob of zeroing PET per paired sample during training "
                        f"(default: {PET_DROPOUT_PROB}). Active only with "
                        f"--use_missing_gate.")
    p.add_argument("--paired_only",        action="store_true", default=False,
                   help="With --use_missing_gate: exclude CT-only patients so the "
                        "gate is trained on paired CT+PET data only.")
    p.add_argument("--use_ct_finetune",    action="store_true", default=False,
                   help="After main training, fine-tune CT branch + head_ct on "
                        "CT-only patients. PET branch and head_fused stay frozen. "
                        "Requires --use_missing_gate.")
    p.add_argument("--paired_only_val",    action="store_true", default=False,
                   help="With --use_missing_gate: always validate on paired patients "
                        "only, with fold splits generated on paired patients. "
                        "CT-only patients are added to training folds only. "
                        "Enables fair comparison with paired_only experiments.")
    p.add_argument("--eval_drop_pet",      action="store_true", default=False,
                   help="After training, additionally evaluate on the paired val set "
                        "with PET zeroed out (CT-only ablation on paired patients). "
                        "Requires --use_missing_gate for gate to suppress zeroed PET.")

    # ── Global CT+Dose branch ─────────────────────────────────────────────────
    p.add_argument("--use_global_branch", action="store_true", default=False,
                   help="Add a global CT+Dose branch fused via gated fusion. "
                        "Full CT and dose volumes are resized to GLOBAL_SIZE "
                        "and fed through a lightweight encoder. Requires --dose_dir.")

    # ── BG-patch MIL mode ─────────────────────────────────────────────────────
    p.add_argument("--use_bg_patches",         action="store_true", default=False,
                   help="Enable slice-level tumor+background paired sampling "
                        "(requires --use_mil). Each bag = K slices × 2 patches.")
    p.add_argument("--n_slices",               type=int,   default=5,
                   help="K: number of slices sampled per patient in bg-patch mode "
                        "(default: 5 → 10 patches per bag).")
    p.add_argument("--bg_min_dist_mm",         type=float, default=50.0,
                   help="Min physical distance (mm) from tumor mask for bg patch "
                        "center (default: 50).")
    # ── Attention regularisation ──────────────────────────────────────────────
    p.add_argument("--attn_sparsity_weight",   type=float, default=0.0,
                   help="Weight for attention entropy-minimisation loss (default: 0).")
    p.add_argument("--rank_constraint_weight", type=float, default=0.0,
                   help="Weight for tumor>bg attention rank constraint (default: 0).")
    p.add_argument("--rank_margin",            type=float, default=0.1,
                   help="Hinge margin for rank constraint loss (default: 0.1).")

    # ── Auxiliary CT loss ─────────────────────────────────────────────────────
    p.add_argument("--ct_aux_weight", type=float, default=0.0,
                   help="Weight for auxiliary CT-only head loss on all patients. "
                        "Forces CT branch to maintain independent predictive features. "
                        "Active only with --use_missing_gate. Default 0 (disabled).")

    # ── Unipair mode ──────────────────────────────────────────────────────────
    p.add_argument("--use_unipair", action="store_true", default=False,
                   help="Treat each modality as independent patient: CT and PET from "
                        "paired patients become separate samples with modality flag. "
                        "Model has separate heads for CT (head_ct) and PET (head_pet). "
                        "Mutually exclusive with --use_missing_gate, --use_ct_only, "
                        "--use_pet_only, --use_bg_patches.")
    p.add_argument("--use_naive_joint", action="store_true", default=False,
                   help="Naive joint training: CT-only patients are included with zero "
                        "PET through the standard single-head classifier. No gate, no "
                        "separate heads. Use with --paired_only_val so val is paired-only. "
                        "Mutually exclusive with --use_missing_gate, --use_unipair, "
                        "--use_ct_only, --use_pet_only.")

    return p.parse_args()


def main():
    rank, local_rank, world_size = setup_ddp()
    is_main = (rank == 0)

    args = parse_args()
    assert not args.use_bg_patches or args.use_mil, \
        "--use_bg_patches requires --use_mil to be set."
    assert not (args.use_missing_gate and args.use_ct_only), \
        "--use_missing_gate and --use_ct_only are mutually exclusive."
    assert not (args.use_pet_only and args.use_ct_only), \
        "--use_pet_only and --use_ct_only are mutually exclusive."
    assert not (args.use_pet_only and args.use_missing_gate), \
        "--use_pet_only and --use_missing_gate are mutually exclusive."
    assert not (args.use_pet_only and args.use_dual_scale), \
        "--use_pet_only and --use_dual_scale are mutually exclusive."
    assert not (args.use_missing_gate and args.use_unipair), \
        "--use_missing_gate and --use_unipair are mutually exclusive."
    assert not (args.use_naive_joint and args.use_missing_gate), \
        "--use_naive_joint and --use_missing_gate are mutually exclusive."
    assert not (args.use_naive_joint and args.use_unipair), \
        "--use_naive_joint and --use_unipair are mutually exclusive."
    assert not (args.use_naive_joint and args.use_ct_only), \
        "--use_naive_joint and --use_ct_only are mutually exclusive."
    assert not (args.use_naive_joint and args.use_pet_only), \
        "--use_naive_joint and --use_pet_only are mutually exclusive."
    set_seed(args.seed + rank)   # different seed per rank → diverse augmentation
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)

    pet_dir = args.pet_dir if args.pet_dir else args.ct_dir
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # ── Model size selection ──────────────────────────────────────────────────
    if args.model_size == "small":
        _dla_channels  = DLA_CHANNELS_SMALL
        _fusion_hidden = FUSION_HIDDEN_SMALL
        _dose_channels = DOSE_CHANNELS_SMALL
    else:
        _dla_channels  = DLA_CHANNELS
        _fusion_hidden = FUSION_HIDDEN
        _dose_channels = DOSE_CHANNELS

    if is_main:
        print(f"Device     : {device}  (world_size={world_size})")
        print(f"Model size : {args.model_size}  "
              f"(DLA channels: {_dla_channels}, fusion_hidden: {_fusion_hidden})")
        print(f"CT dir : {args.ct_dir}")
        if not args.use_ct_only:
            print(f"PET dir: {pet_dir}")
        print(f"CSV    : {args.csv}")
        print(f"Label  : {args.label_col}")
        if args.use_ct_only:
            print(f"Mode   : CT-only ablation  (MIL={'on' if args.use_mil else 'off'})")

    # ── Build full patient list from CSV ──────────────────────────────────────
    df  = pd.read_csv(args.csv)
    df  = parse_labels(df, args.label_col)
    ids = df['Patient_ID'].astype(str).tolist()
    lbs = df['label'].astype(int).tolist()
    if is_main:
        print(f"Total patients: {len(ids)}  pos={sum(lbs)}  neg={len(lbs)-sum(lbs)}")

    # ── Stratified K-Fold ─────────────────────────────────────────────────────
    skf     = StratifiedKFold(n_splits=args.num_folds, shuffle=True,
                               random_state=args.seed)
    ids_arr = np.array(ids)
    lbs_arr = np.array(lbs)

    # paired_only_val (or paired_only) mode:
    #   Fold splits are generated on PAIRED patients only.
    #   CT-only patients are appended to every training fold.
    #   Validation folds contain only paired patients → identical to paired_only
    #   experiment for fair comparison.
    #
    # Plain missing_gate (no paired_only_val):
    #   Stratify on 4 buckets (is_paired × label) so CT-only ratio stays
    #   consistent across folds.
    #
    # All other modes: standard 2-class stratification.

    _use_paired_splits = (
        (args.use_missing_gate or args.use_unipair or args.use_naive_joint)
        and not args.use_ct_only
        and (args.paired_only_val or args.paired_only)
    )

    if (args.use_missing_gate or args.use_unipair or args.use_naive_joint) and not args.use_ct_only:
        paired_arr = np.array([
            int(os.path.exists(os.path.join(pet_dir, f"{pid}_PET.npy")))
            for pid in ids
        ])
        if is_main and args.use_missing_gate:
            for key, name in [(0,"CT-only,neg"),(1,"CT-only,pos"),
                              (2,"paired,neg"), (3,"paired,pos")]:
                bucket = (paired_arr * 2 + lbs_arr == key).sum()
                print(f"  strat bucket [{name}]: {bucket}")
        if is_main and (args.use_unipair or args.use_naive_joint):
            n_paired = int(paired_arr.sum())
            n_ctonly = len(paired_arr) - n_paired
            mode_tag = "unipair" if args.use_unipair else "naive_joint"
            print(f"  {mode_tag} splits: {n_paired} paired + {n_ctonly} CT-only patients")
    else:
        paired_arr = np.ones(len(ids), dtype=int)   # all paired in non-mg modes

    if _use_paired_splits:
        # Build paired-only index arrays for fold generation
        paired_mask     = paired_arr.astype(bool)
        p_ids_arr       = ids_arr[paired_mask]
        p_lbs_arr       = lbs_arr[paired_mask]
        ctonly_ids_list = ids_arr[~paired_mask].tolist()
        fold_iter       = skf.split(p_ids_arr, p_lbs_arr)
        # Also split CT-only patients across folds so val has no leakage
        if ctonly_ids_list:
            c_ids_arr  = np.array(ctonly_ids_list)
            c_lbs_arr  = np.array([
                int(lbs_arr[np.where(ids_arr == pid)[0][0]])
                for pid in ctonly_ids_list
            ])
            ctonly_fold_splits = list(skf.split(c_ids_arr, c_lbs_arr))
        else:
            c_ids_arr          = np.array([])
            ctonly_fold_splits = None
        if is_main and ctonly_ids_list:
            print(f"  paired_only_val: {len(p_ids_arr)} paired + "
                  f"{len(ctonly_ids_list)} CT-only patients, both split across folds.")
    else:
        if args.use_missing_gate and not args.use_ct_only:
            strat_arr = paired_arr * 2 + lbs_arr
        else:
            strat_arr = lbs_arr
        fold_iter = skf.split(ids_arr, strat_arr)
        p_ids_arr = ids_arr
        ctonly_ids_list = []

    fold_results = []

    # CT-only patient list for fine-tune phase (all CT-only patients, not split by fold)
    # They all go into training in every fold's fine-tune phase.
    # Works regardless of use_missing_gate: fine-tune only needs CT files.
    if args.use_ct_finetune:
        if not ctonly_ids_list:
            _all_paired = np.array([
                int(os.path.exists(os.path.join(pet_dir, f"{pid}_PET.npy")))
                for pid in ids
            ])
            ctonly_ids_list = ids_arr[~_all_paired.astype(bool)].tolist()
        if is_main:
            print(f"CT fine-tune: {len(ctonly_ids_list)} CT-only patients available.")

    common_ds_kwargs = dict(
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
        use_mil         = args.use_mil,
        patch_size      = tuple(args.patch_size),
        mil_patch_size  = tuple(args.mil_patch_size),
        mil_offsets     = MIL_OFFSETS,
        use_dual_scale  = args.use_dual_scale,
        large_patch_size = LARGE_PATCH_SIZE,
        use_ct_only       = args.use_ct_only,
        use_pet_only      = args.use_pet_only,
        use_bg_patches    = args.use_bg_patches,
        n_slices          = args.n_slices,
        bg_min_dist_mm    = args.bg_min_dist_mm,
        use_missing_gate  = args.use_missing_gate,
        paired_only       = args.paired_only,
        pet_dropout_prob  = args.pet_dropout_prob,
        use_global_branch = args.use_global_branch,
        global_size       = GLOBAL_SIZE,
        use_unipair       = args.use_unipair,
        use_naive_joint   = args.use_naive_joint,
    )

    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter, start=1):

        if args.folds is not None and fold_idx not in args.folds:
            if is_main:
                print(f"\nSkipping fold {fold_idx} (not in --folds {args.folds})")
            continue

        if _use_paired_splits:
            # train: paired train fold + CT-only train fold
            # val:   paired val fold  (CT-only val fold handled separately below)
            if ctonly_fold_splits is not None:
                ct_tr_idx, ct_val_idx = ctonly_fold_splits[fold_idx - 1]
                ctonly_train_ids_fold = c_ids_arr[ct_tr_idx].tolist()
                ctonly_val_ids_fold   = c_ids_arr[ct_val_idx].tolist()
            else:
                ctonly_train_ids_fold = []
                ctonly_val_ids_fold   = []
            train_ids = p_ids_arr[train_idx].tolist() + ctonly_train_ids_fold
            val_ids   = p_ids_arr[val_idx].tolist()
        else:
            train_ids = ids_arr[train_idx].tolist()
            val_ids   = ids_arr[val_idx].tolist()

        if is_main:
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx}/{args.num_folds}  "
                  f"train={len(train_ids)}  val={len(val_ids)}")
            print(f"{'='*60}")

        train_ds = PatchDataset(
            **common_ds_kwargs,
            patient_ids = train_ids,
            augment     = True,
            mixup_prob  = MIXUP_PROB,
            mixup_alpha = MIXUP_ALPHA,
            preload     = args.preload,
        )
        # ctonly_ds for fine-tune phase: CT-only patients with use_missing_gate=True
        # so they're included regardless of model's missing_gate setting.
        # Dataset returns 5-tuples (ct, zero_pet, dose, label, pet_present=0).
        ctonly_ds = None
        if args.use_ct_finetune and ctonly_ids_list:
            ctonly_ds = PatchDataset(
                **{**common_ds_kwargs, 'use_missing_gate': True, 'paired_only': False},
                patient_ids = ctonly_ids_list,
                augment     = True,
                preload     = args.preload,
            )

        # val_ds: always paired-only when _use_paired_splits so that
        # missing_gate and paired_only experiments share identical val folds.
        val_ds = PatchDataset(
            **{**common_ds_kwargs,
               'paired_only': _use_paired_splits or args.paired_only},
            patient_ids = val_ids,
            augment     = False,
            preload     = args.preload,
        )

        # ── Train fold ────────────────────────────────────────────────────────
        result = train_fold(
            fold_idx                 = fold_idx,
            train_dataset            = train_ds,
            val_dataset              = val_ds,
            device                   = device,
            save_dir                 = args.out_dir,
            use_dose                 = args.use_dose,
            use_mil                  = args.use_mil,
            dla_channels             = _dla_channels,
            fusion_hidden            = _fusion_hidden,
            dose_channels            = _dose_channels,
            use_dual_scale           = args.use_dual_scale,
            ct_large_channels        = CT_LARGE_CHANNELS,
            dose_large_channels      = DOSE_LARGE_CHANNELS,
            dual_scale_fusion_hidden = DUAL_SCALE_FUSION_HIDDEN,
            large_branch_use_ct      = DUAL_SCALE_USE_CT_LARGE,
            use_ct_only              = args.use_ct_only,
            use_pet_only             = args.use_pet_only,
            use_bg_patches           = args.use_bg_patches,
            attn_sparsity_weight     = args.attn_sparsity_weight,
            rank_constraint_weight   = args.rank_constraint_weight,
            rank_margin              = args.rank_margin,
            use_missing_gate         = args.use_missing_gate,
            use_global_branch        = args.use_global_branch,
            use_ct_finetune          = args.use_ct_finetune,
            ctonly_dataset           = ctonly_ds,
            eval_drop_pet            = args.eval_drop_pet,
            ct_aux_weight            = args.ct_aux_weight,
            use_unipair              = args.use_unipair,
            rank                     = rank,
            world_size               = world_size,
            num_workers              = args.num_workers,
        )

        # ── Post-fold evaluation (rank 0 only) ────────────────────────────────
        # rank 0 runs inference; non-main ranks wait at the barrier below.
        if is_main:
            # ── Helper: load a checkpoint with the same model config ──────────
            def _load_fold_model(ckpt_path):
                return load_model(
                    ckpt_path, device,
                    use_dose=args.use_dose, use_mil=args.use_mil,
                    dla_channels=_dla_channels, fusion_hidden=_fusion_hidden,
                    dose_channels=_dose_channels,
                    use_dual_scale=args.use_dual_scale,
                    ct_large_channels=CT_LARGE_CHANNELS,
                    dose_large_channels=DOSE_LARGE_CHANNELS,
                    dual_scale_fusion_hidden=DUAL_SCALE_FUSION_HIDDEN,
                    large_branch_use_ct=DUAL_SCALE_USE_CT_LARGE,
                    use_ct_only=args.use_ct_only,
                    use_pet_only=args.use_pet_only,
                    use_missing_gate=args.use_missing_gate,
                    use_unipair=args.use_unipair,
                )

            # ── Build DataLoaders ─────────────────────────────────────────────
            train_eval_ds = PatchDataset(
                **common_ds_kwargs,
                patient_ids = train_ids,
                augment     = False,
            )
            train_loader = DataLoader(
                train_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
            )

            # CT-only val loader: held-out CT-only fold + paired val with PET zeroed.
            # Both groups are evaluated with zero_pet=True so the model only uses CT.
            ctonly_val_loader = None
            _ctonly_combined_ids = []
            if args.use_missing_gate:
                if _use_paired_splits:
                    # CT-only val = CT-only fold val + paired val (PET will be zeroed)
                    _ctonly_combined_ids = ctonly_val_ids_fold + val_ids
                else:
                    # Non-paired-splits: CT-only subset of this fold's val ids only
                    _ctonly_set = set(ctonly_ids_list)
                    _ctonly_combined_ids = [p for p in val_ids if p in _ctonly_set]
                if _ctonly_combined_ids:
                    _ctonly_val_ds = PatchDataset(
                        **{**common_ds_kwargs,
                           'use_missing_gate': True, 'paired_only': False},
                        patient_ids = _ctonly_combined_ids,
                        augment     = False,
                    )
                    ctonly_val_loader = DataLoader(
                        _ctonly_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                    )
                    if is_main:
                        print(f"  CT-only val: {len(ctonly_val_ids_fold)} CT-only "
                              f"+ {len(val_ids)} paired (PET zeroed) "
                              f"= {len(_ctonly_combined_ids)} total")

            # ── Helper: evaluate one model on paired-val + CT-only-val ────────
            def _eval_fold_model(mdl, tag):
                """Returns (thr, metrics_paired, metrics_ctonly,
                            yt_val, yp_val, pet_arr_val)."""
                # Threshold from training split (Youden index)
                yt_tr, yp_tr, _ = run_inference(
                    mdl, train_loader, device,
                    use_dual_scale=args.use_dual_scale,
                    use_ct_only=args.use_ct_only,
                    use_pet_only=args.use_pet_only,
                    use_missing_gate=args.use_missing_gate,
                    use_global_branch=args.use_global_branch,
                    use_unipair=args.use_unipair,
                )
                thr = find_optimal_threshold(yt_tr, yp_tr)
                print(f"  [{tag}] Threshold (Youden on train): {thr:.3f}")

                # ── Val (paired, or all patients in pet_only/ct_only mode) ───
                yt_val, yp_val, pet_arr_val = run_inference(
                    mdl, val_loader, device,
                    use_dual_scale=args.use_dual_scale,
                    use_ct_only=args.use_ct_only,
                    use_pet_only=args.use_pet_only,
                    use_missing_gate=args.use_missing_gate,
                    use_global_branch=args.use_global_branch,
                    use_unipair=args.use_unipair,
                )
                m_paired = compute_metrics(yt_val, yp_val, threshold=thr)
                print(f"  [{tag}] Val (n={len(yt_val)}, threshold={thr:.3f}):")
                for k, v in m_paired.items():
                    print(f"    {k:<14s}: {v}")

                # Per-modality full metrics in unipair mode
                if args.use_unipair and pet_arr_val is not None:
                    for key, mask_val in [("CT",  pet_arr_val == 0),
                                           ("PET", pet_arr_val == 1)]:
                        mask_val = mask_val.astype(bool)
                        if mask_val.sum() >= 2 and len(np.unique(yt_val[mask_val])) > 1:
                            m_mod = compute_metrics(yt_val[mask_val], yp_val[mask_val],
                                                    threshold=thr)
                            print(f"  [{tag}] {key}-only (n={mask_val.sum()}, "
                                  f"threshold={thr:.3f}):")
                            for k, v in m_mod.items():
                                print(f"    {k:<14s}: {v}")

                # ── PET-only ablation (zero-out CT) ───────────────────────────
                if not args.use_ct_only and not args.use_pet_only and not args.use_dual_scale and not args.use_unipair:
                    yt_pet, yp_pet, _ = run_inference(
                        mdl, val_loader, device,
                        use_dual_scale=False,
                        use_ct_only=False,
                        use_missing_gate=args.use_missing_gate,
                        use_global_branch=False,
                        zero_ct=True,
                    )
                    if len(yt_pet) >= 2 and len(np.unique(yt_pet)) > 1:
                        m_pet = compute_metrics(yt_pet, yp_pet, threshold=thr)
                        print(f"  [{tag}] PET-only ablation (CT zeroed, "
                              f"n={len(yt_pet)}):")
                        for k, v in m_pet.items():
                            print(f"    {k:<14s}: {v}")

                # ── CT-only ablation (zero-out PET on paired val set) ─────────
                m_drop_pet = None
                if (args.eval_drop_pet
                        and not args.use_ct_only
                        and not args.use_pet_only
                        and not args.use_dual_scale
                        and not args.use_unipair):
                    yt_ctabl, yp_ctabl, _ = run_inference(
                        mdl, val_loader, device,
                        use_dual_scale=False,
                        use_ct_only=False,
                        use_missing_gate=args.use_missing_gate,
                        use_global_branch=False,
                        zero_pet=True,
                    )
                    if len(yt_ctabl) >= 2 and len(np.unique(yt_ctabl)) > 1:
                        m_drop_pet = compute_metrics(yt_ctabl, yp_ctabl, threshold=thr)
                        print(f"  [{tag}] CT-only ablation (PET zeroed, "
                              f"n={len(yt_ctabl)}):")
                        for k, v in m_drop_pet.items():
                            print(f"    {k:<14s}: {v}")

                # ── CT-only val ───────────────────────────────────────────────
                m_ctonly = None
                if ctonly_val_loader is not None:
                    yt_ct, yp_ct, _ = run_inference(
                        mdl, ctonly_val_loader, device,
                        use_dual_scale=args.use_dual_scale,
                        use_ct_only=args.use_ct_only,
                        use_missing_gate=True,   # ctonly_val_ds is always MG-format
                        use_global_branch=args.use_global_branch,
                        zero_pet=True,           # force CT-only: zero PET for all
                    )
                    if len(yt_ct) >= 2 and len(np.unique(yt_ct)) > 1:
                        m_ctonly = compute_metrics(yt_ct, yp_ct, threshold=thr)
                        print(f"  [{tag}] CT-only val (n={len(yt_ct)}):")
                        for k, v in m_ctonly.items():
                            print(f"    {k:<14s}: {v}")
                    else:
                        print(f"  [{tag}] CT-only val: n={len(yt_ct)}, "
                              f"skipping metrics (insufficient samples).")

                return thr, m_paired, m_ctonly, m_drop_pet, yt_val, yp_val, pet_arr_val

            # ── Phase 1 model evaluation ──────────────────────────────────────
            print(f"\n{'─'*60}")
            print(f"  [Phase 1 — fold {fold_idx}]")
            model = _load_fold_model(result["model_path"])
            thr, final_metrics, _, m_drop_pet, yt_val, yp_val, pet_arr_val = \
                _eval_fold_model(model, f"Phase1/Fold{fold_idx}")

            # Per-group AUCs when the full val set contains mixed types
            if args.use_missing_gate and pet_arr_val is not None:
                from sklearn.metrics import roc_auc_score
                for key, mask in [("paired", pet_arr_val == 1),
                                   ("ctonly", pet_arr_val == 0)]:
                    mask = mask.astype(bool)
                    if mask.sum() >= 2 and len(np.unique(yt_val[mask])) > 1:
                        final_metrics[f"auc_{key}"] = float(
                            roc_auc_score(yt_val[mask], yp_val[mask]))
                    else:
                        final_metrics[f"auc_{key}"] = float("nan")
            # Per-modality AUCs for unipair mode
            if args.use_unipair and pet_arr_val is not None:
                from sklearn.metrics import roc_auc_score
                for key, mask in [("ct",  pet_arr_val == 0),
                                   ("pet", pet_arr_val == 1)]:
                    mask = mask.astype(bool)
                    if mask.sum() >= 2 and len(np.unique(yt_val[mask])) > 1:
                        final_metrics[f"auc_{key}"] = float(
                            roc_auc_score(yt_val[mask], yp_val[mask]))
                    else:
                        final_metrics[f"auc_{key}"] = float("nan")
            result["val_metrics"]          = final_metrics
            result["val_metrics_drop_pet"] = m_drop_pet
            result["best_threshold"]       = thr

            # ── Attention weight analysis (bg-patch MIL only) ─────────────────
            if args.use_mil and args.use_bg_patches:
                tumor_attns, bg_attns = [], []
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        ct_b   = batch[0].to(device)
                        pet_b  = batch[1].to(device)
                        dose_b = batch[2].to(device)
                        ptypes = batch[4]          # (B, 2K) 0=tumor / 1=bg
                        pp_b = batch[-1].to(device) if args.use_missing_gate else None
                        _, A = model(ct_b, pet_b, dose_b, return_attn=True,
                                     pet_present=pp_b)
                        A_sq = A.squeeze(-1)       # (B, 2K)
                        for i in range(A_sq.shape[0]):
                            t_mask = (ptypes[i] == 0)
                            b_mask = (ptypes[i] == 1)
                            if t_mask.any():
                                tumor_attns.append(A_sq[i][t_mask].mean().item())
                            if b_mask.any():
                                bg_attns.append(A_sq[i][b_mask].mean().item())
                mean_t = float(np.mean(tumor_attns)) if tumor_attns else float('nan')
                mean_b = float(np.mean(bg_attns))    if bg_attns    else float('nan')
                ratio  = mean_t / (mean_b + 1e-8)
                print(f"  Attention (val fold {fold_idx}) — "
                      f"tumor: {mean_t:.4f}  bg: {mean_b:.4f}  ratio: {ratio:.2f}x")

            plot_roc(
                yt_val, yp_val, thr,
                save_path = os.path.join(args.out_dir, f"fold{fold_idx}_roc.png"),
                title     = f"Fold {fold_idx} Validation ROC (Phase 1)",
            )

            # ── Fine-tuned model evaluation ───────────────────────────────────
            if args.use_ct_finetune and result.get("ft_model_path"):
                print(f"\n{'─'*60}")
                print(f"  [Fine-tuned — fold {fold_idx}]")
                ft_model = _load_fold_model(result["ft_model_path"])
                ft_thr, ft_m_paired, ft_m_ctonly, ft_m_drop_pet, ft_yt_val, ft_yp_val, _ = \
                    _eval_fold_model(ft_model, f"FineTune/Fold{fold_idx}")
                result["ft_val_metrics"]             = ft_m_paired
                result["ft_val_metrics_ctonly"]      = ft_m_ctonly
                result["ft_val_metrics_drop_pet"]    = ft_m_drop_pet
                result["ft_threshold"]               = ft_thr
                plot_roc(
                    ft_yt_val, ft_yp_val, ft_thr,
                    save_path = os.path.join(args.out_dir,
                                             f"fold{fold_idx}_ft_roc.png"),
                    title     = f"Fold {fold_idx} Validation ROC (Fine-tuned)",
                )

            fold_results.append(result)

        # All ranks sync here so non-main ranks don't race into the next fold's
        # DDP collectives while rank 0 is still doing post-fold inference.
        if dist.is_initialized():
            dist.barrier()

    if is_main:
        print_summary(fold_results, out_dir=args.out_dir)

    cleanup_ddp()


if __name__ == "__main__":
    main()
