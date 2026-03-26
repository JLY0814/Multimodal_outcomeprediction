#!/usr/bin/env bash
# run.sh — Dual-branch 3D CNN (CT + PET) cross-validation training + evaluation
# Usage: bash run.sh [recurrence|figo] [base|small] [num_gpus]
#
# All stdout + stderr are saved to $OUTPUT_DIR/train.log (shown in terminal too).

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"

# ── Dose (optional third modality for tri-branch model) ───────────────────────
# Set USE_DOSE=true to add Dose as a third small-patch branch (same size as CT/PET).
USE_DOSE=false
DOSE_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"  # {Patient_ID}_Dose.npy

# ── MIL-Attention (optional, compatible with USE_DOSE) ────────────────────────
# 7 instances by default (centre + 6 offsets); set true to enable.
USE_MIL=false

# ── Dual-scale (Branch1: CT+PET small patch, Branch2: CT+Dose large patch) ────
# Requires DOSE_DIR to be set above. USE_DOSE can stay false.
USE_DUAL_SCALE=false

# ── CT-only ablation (no PET, no Dose; compatible with USE_MIL) ───────────────
# Set true to run single-branch CT-only model. PET dir is not needed.
USE_CT_ONLY=false

# ── PET-only ablation (no CT, no Dose; compatible with USE_MIL) ──────────────
# Set true to run single-branch PET-only model. CT dir is not needed.
USE_PET_ONLY=false

# ── BG-patch MIL mode (requires USE_MIL=true) ────────────────────────────────
# Each bag = K slices × 2 patches (1 tumor + 1 background).
# Attention regularisation weights default to 0 (no reg if not set).
USE_BG_PATCHES=false

# ── Unipair mode: treat each modality as independent patient ──────────────────
# Paired CT+PET patients each contribute 2 training samples:
#   Sample A: (ct_patch, zeros_pet, label, modality=0)  ← CT-only sample
#   Sample B: (zeros_ct, pet_patch, label, modality=1)  ← PET-only sample
# Model has separate heads for CT (head_ct) and PET (head_pet).
# Mutually exclusive with USE_MISSING_GATE, USE_CT_ONLY, USE_PET_ONLY, USE_BG_PATCHES.
USE_UNIPAIR=false

# ── Global CT+Dose branch ─────────────────────────────────────────────────────
# Full-volume CT modulated by dose map; processed by a lightweight global encoder.
# Fused with the ROI token via a learned scalar gate. Requires --dose_dir.
USE_GLOBAL_BRANCH=false

# ── Missing modality gate (CT+PET unified loop, PET optional) ─────────────────
# Set USE_MISSING_GATE=true to train a gate-equipped dual-branch model that
# accepts both paired CT+PET patients AND CT-only patients in the same loop.
# Patients without a PET file are automatically treated as CT-only (pet_present=0).
# PET_DROPOUT_PROB randomly zeros PET for paired samples during training (robustness).
USE_MISSING_GATE=false
PET_DROPOUT_PROB=0.25
CT_AUX_WEIGHT=0.0       # auxiliary CT-only loss weight (0=disabled, e.g. 0.3)
PAIRED_ONLY=false        # with USE_MISSING_GATE: exclude CT-only patients (gate + paired-only)
PAIRED_ONLY_VAL=false   # with USE_MISSING_GATE: fold splits on paired patients; CT-only → train only; val always paired
USE_CT_FINETUNE=false   # after main training, fine-tune CT branch + head_ct on CT-only patients
EVAL_DROP_PET=false     # post-training: re-evaluate val set with PET zeroed (CT-only robustness ablation)

# ── Label to predict: 'recurrence' or 'figo' (pass as first argument) ─────────
LABEL_COL="${1:-recurrence}"

# ── Model size: 'base' (~26M params) or 'small' (~6M params) ─────────────────
MODEL_SIZE="${2:-base}"

# ── Number of GPUs (pass as third argument, default 1) ────────────────────────
# e.g.  bash run.sh recurrence base 4   → 4-GPU DDP via torchrun
NUM_GPUS="${3:-1}"

# ── GPU device selection ───────────────────────────────────────────────────────
# Set to comma-separated GPU IDs to use specific devices, e.g. "0,1" or "4,5".
# Leave empty ("") to use all visible GPUs (default behaviour).
# Run `nvidia-smi --query-gpu=index,memory.free --format=csv` to see free GPUs.
GPUS=""

# ── Fold selection (optional) ─────────────────────────────────────────────────
# Leave empty ("") to run all 5 folds.
# Set to space-separated fold numbers (1-based) to run only those folds.
# e.g. FOLDS="3 4"  →  only run fold 3 and fold 4
FOLDS=""

# ── Output directory (timestamped, name reflects mode) ────────────────────────
if [ "$USE_DUAL_SCALE" = "true" ]; then
    _MODE_TAG="dual_scale"
elif [ "$USE_CT_ONLY" = "true" ]; then
    _MODE_TAG="ct_only"
elif [ "$USE_PET_ONLY" = "true" ]; then
    _MODE_TAG="pet_only"
elif [ "$USE_MISSING_GATE" = "true" ]; then
    _MODE_TAG="missing_gate"
elif [ "$USE_UNIPAIR" = "true" ]; then
    _MODE_TAG="unipair"
elif [ "$USE_NAIVE_JOINT" = "true" ]; then
    _MODE_TAG="naive_joint"
else
    _MODE_TAG="dual_branch"
fi
# Append _mil suffix when MIL is enabled (works for any base mode)
if [ "$USE_MIL" = "true" ]; then
    _MODE_TAG="${_MODE_TAG}_mil"
fi
OUTPUT_DIR="./outputs/${_MODE_TAG}_${LABEL_COL}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/train.log"

# ── Redirect all subsequent output (stdout + stderr) to log + terminal ─────────
exec > >(tee -a "$LOG_FILE") 2>&1

# Disable Python output buffering so every print appears in the log immediately
export PYTHONUNBUFFERED=1

echo "============================================================"
if [ "$USE_CT_ONLY" = "true" ]; then
    echo "  CT-only ablation (no PET / Dose)"
else
    echo "  Dual-branch 3D CNN — CT + PET"
fi
echo "  Started   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CSV       : $CSV"
echo "  CT dir    : $CT_DIR"
if [ "$USE_CT_ONLY" != "true" ]; then
    echo "  PET dir   : $PET_DIR"
fi
echo "  Label     : $LABEL_COL"
echo "  Mask dir  : $MASK_DIR"
echo "  MIL          : $USE_MIL"
echo "  CT-only      : $USE_CT_ONLY"
echo "  Missing gate : $USE_MISSING_GATE  (PET dropout: $PET_DROPOUT_PROB)"
echo "  Output       : $OUTPUT_DIR"
echo "  Log       : $LOG_FILE"
echo "============================================================"

# ── 5-fold cross-validation training ─────────────────────────────────────────
# ct_axis_order  ZYX : CT stored as (Z,Y,X) → code transposes to (X,Y,Z)
# pet_axis_order XYZ : PET already in "normal" (X,Y,Z) order → no transpose
# Build dose flags dynamically
DOSE_FLAGS=""
if [ "$USE_DOSE" = "true" ]; then
    DOSE_FLAGS="--use_dose --dose_dir $DOSE_DIR --dose_axis_order ZYX --dose_max 1"
fi

MIL_FLAGS=""
if [ "$USE_MIL" = "true" ]; then
    MIL_FLAGS="--use_mil"
fi

DUAL_SCALE_FLAGS=""
if [ "$USE_DUAL_SCALE" = "true" ]; then
    # dose_dir is required for the large-patch Dose branch even when USE_DOSE=false
    DUAL_SCALE_FLAGS="--use_dual_scale --dose_dir $DOSE_DIR --dose_axis_order ZYX --dose_max 1"
fi

CT_ONLY_FLAGS=""
if [ "$USE_CT_ONLY" = "true" ]; then
    CT_ONLY_FLAGS="--use_ct_only"
fi

PET_ONLY_FLAGS=""
if [ "$USE_PET_ONLY" = "true" ]; then
    PET_ONLY_FLAGS="--use_pet_only"
fi

BG_FLAGS=""
if [ "$USE_BG_PATCHES" = "true" ]; then
    BG_FLAGS="--use_bg_patches --n_slices 5 --bg_min_dist_mm 50 \
              --attn_sparsity_weight 0.01 --rank_constraint_weight 0.05 \
              --rank_margin 0.1"
fi

UNIPAIR_FLAGS=""
if [ "$USE_UNIPAIR" = "true" ]; then
    UNIPAIR_FLAGS="--use_unipair"
fi

NAIVE_JOINT_FLAGS=""
if [ "$USE_NAIVE_JOINT" = "true" ]; then
    NAIVE_JOINT_FLAGS="--use_naive_joint"
    if [ "$PAIRED_ONLY_VAL" = "true" ]; then
        NAIVE_JOINT_FLAGS="$NAIVE_JOINT_FLAGS --paired_only_val"
    fi
fi

MISSING_GATE_FLAGS=""
if [ "$USE_MISSING_GATE" = "true" ]; then
    MISSING_GATE_FLAGS="--use_missing_gate --pet_dropout_prob $PET_DROPOUT_PROB"
    if [ "$PAIRED_ONLY" = "true" ]; then
        MISSING_GATE_FLAGS="$MISSING_GATE_FLAGS --paired_only"
    fi
    if [ "$PAIRED_ONLY_VAL" = "true" ]; then
        MISSING_GATE_FLAGS="$MISSING_GATE_FLAGS --paired_only_val"
    fi
    if [ "$(echo "$CT_AUX_WEIGHT > 0" | bc -l)" = "1" ]; then
        MISSING_GATE_FLAGS="$MISSING_GATE_FLAGS --ct_aux_weight $CT_AUX_WEIGHT"
    fi
fi

CT_FINETUNE_FLAGS=""
if [ "$USE_CT_FINETUNE" = "true" ]; then
    CT_FINETUNE_FLAGS="--use_ct_finetune"
fi

EVAL_DROP_PET_FLAGS=""
if [ "$EVAL_DROP_PET" = "true" ]; then
    EVAL_DROP_PET_FLAGS="--eval_drop_pet"
fi

GLOBAL_BRANCH_FLAGS=""
if [ "$USE_GLOBAL_BRANCH" = "true" ]; then
    GLOBAL_BRANCH_FLAGS="--use_global_branch --dose_dir $DOSE_DIR --dose_axis_order ZYX --dose_max 1"
fi

# ── GPU selection (set GPUS to the comma-separated device IDs you want) ───────
GPUS="${GPUS:-}"                       # e.g. export GPUS=4,5 before running
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# ── Launch: torchrun for multi-GPU, plain python for single GPU ───────────────
MASTER_PORT=$(shuf -i 29500-29999 -n 1)
LAUNCHER="python"
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT"
fi

FOLDS_FLAGS=""
if [ -n "$FOLDS" ]; then
    FOLDS_FLAGS="--folds $FOLDS"
fi

$LAUNCHER main.py \
    --csv            "$CSV"         \
    --ct_dir         "$CT_DIR"      \
    --pet_dir        "$PET_DIR"     \
    --mask_dir       "$MASK_DIR"    \
    --label_col      "$LABEL_COL"  \
    --ct_axis_order  ZYX           \
    --pet_axis_order XYZ           \
    --ct_wl          40            \
    --ct_ww          400           \
    --pet_max        200           \
    --num_folds      5             \
    --seed           42            \
    --num_workers    4             \
    --out_dir        "$OUTPUT_DIR" \
    $DOSE_FLAGS \
    $MIL_FLAGS \
    $DUAL_SCALE_FLAGS \
    $CT_ONLY_FLAGS \
    $PET_ONLY_FLAGS \
    $BG_FLAGS \
    $UNIPAIR_FLAGS \
    $NAIVE_JOINT_FLAGS \
    $MISSING_GATE_FLAGS \
    $CT_FINETUNE_FLAGS \
    $EVAL_DROP_PET_FLAGS \
    $GLOBAL_BRANCH_FLAGS \
    $FOLDS_FLAGS \
    --model_size     "$MODEL_SIZE" \
    "${@:4}"

echo ""
echo "============================================================"
echo "  Training complete : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in        : $OUTPUT_DIR"
echo "  Full log          : $LOG_FILE"
echo "============================================================"

# ── Per-fold standalone evaluation on a held-out test set (optional) ──────────
#
# TEST_CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/test_patients_info.csv"
# BEST_FOLD=1
# python evaluate.py \
#     --checkpoint "$OUTPUT_DIR/fold${BEST_FOLD}_best.pth" \
#     --data_root  "$CT_DIR"     \
#     --label_csv  "$TEST_CSV"   \
#     --out_dir    "$OUTPUT_DIR/test_eval"

# ── TensorBoard (optional) ────────────────────────────────────────────────────
# tensorboard --logdir "$OUTPUT_DIR"
