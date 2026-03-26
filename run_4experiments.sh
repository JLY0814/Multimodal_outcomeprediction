#!/usr/bin/env bash
# run_4experiments.sh
# Run 4 missing-modality experiments in parallel, one per GPU.
#
# Experiment layout:
#   GPU 0 — Exp1: paired-only train          → paired-only val
#   GPU 1 — Exp2: naive joint (zero PET)      → paired-only val
#   GPU 2 — Exp3: missing gate (sep. heads)   → paired-only val
#   GPU 3 — Exp4: synthetic missing PET        → paired-only val
#
# Usage:
#   bash run_4experiments.sh [recurrence|figo] [base|small]
#
# Logs: outputs/exp{1..4}_*/train.log

set -euo pipefail

# ── Shared paths ───────────────────────────────────────────────────────────────
CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"

LABEL_COL="${1:-recurrence}"
MODEL_SIZE="${2:-base}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Common python args (shared by all 4 experiments) ──────────────────────────
COMMON_ARGS="
    --csv            $CSV
    --ct_dir         $CT_DIR
    --pet_dir        $PET_DIR
    --mask_dir       $MASK_DIR
    --label_col      $LABEL_COL
    --ct_axis_order  ZYX
    --pet_axis_order XYZ
    --ct_wl          40
    --ct_ww          400
    --pet_max        200
    --num_folds      5
    --seed           42
    --num_workers    4
    --model_size     $MODEL_SIZE
    --use_mil
    --paired_only_val
"

# ── Output dirs ────────────────────────────────────────────────────────────────
OUT1="./outputs/exp1_paired_only_${LABEL_COL}_${TIMESTAMP}"
OUT2="./outputs/exp2_naive_joint_${LABEL_COL}_${TIMESTAMP}"
OUT3="./outputs/exp3_missing_gate_${LABEL_COL}_${TIMESTAMP}"
OUT4="./outputs/exp4_synthetic_missing_${LABEL_COL}_${TIMESTAMP}"

mkdir -p "$OUT1" "$OUT2" "$OUT3" "$OUT4"

echo "============================================================"
echo "  4-Experiment parallel run"
echo "  Label     : $LABEL_COL"
echo "  Model size: $MODEL_SIZE"
echo "  Timestamp : $TIMESTAMP"
echo "  Exp1 → $OUT1"
echo "  Exp2 → $OUT2"
echo "  Exp3 → $OUT3"
echo "  Exp4 → $OUT4"
echo "============================================================"

# ── Exp 1: paired-only ────────────────────────────────────────────────────────
# Standard dual-branch, only paired CT+PET patients in both train and val.
# No CT-only patients, no gate, no dropout.
(
  export CUDA_VISIBLE_DEVICES=0
  export PYTHONUNBUFFERED=1
  echo "[Exp1] Starting on GPU 0 → $OUT1"
  python main.py $COMMON_ARGS \
      --out_dir "$OUT1" \
      --use_missing_gate \
      --paired_only \
      --pet_dropout_prob 0 \
      > "$OUT1/train.log" 2>&1
  echo "[Exp1] Done. Log: $OUT1/train.log"
) &
PID1=$!

# ── Exp 2: naive joint ────────────────────────────────────────────────────────
# CT-only patients join training with zero PET through the standard single
# classifier head. No gate, no separate heads.
(
  export CUDA_VISIBLE_DEVICES=1
  export PYTHONUNBUFFERED=1
  echo "[Exp2] Starting on GPU 1 → $OUT2"
  python main.py $COMMON_ARGS \
      --out_dir "$OUT2" \
      --use_naive_joint \
      > "$OUT2/train.log" 2>&1
  echo "[Exp2] Done. Log: $OUT2/train.log"
) &
PID2=$!

# ── Exp 3: missing gate (separate heads) ──────────────────────────────────────
# CT-only patients train alongside paired patients. Missing-aware gate suppresses
# PET for CT-only samples. Separate head_fused (paired) and head_ct (CT-only).
(
  export CUDA_VISIBLE_DEVICES=2
  export PYTHONUNBUFFERED=1
  echo "[Exp3] Starting on GPU 2 → $OUT3"
  python main.py $COMMON_ARGS \
      --out_dir "$OUT3" \
      --use_missing_gate \
      --pet_dropout_prob 0 \
      > "$OUT3/train.log" 2>&1
  echo "[Exp3] Done. Log: $OUT3/train.log"
) &
PID3=$!

# ── Exp 4: synthetic missing PET on paired only ───────────────────────────────
# Only paired patients, but PET is randomly zeroed during training (prob=0.3).
# Gate learns to rely on CT alone when PET is absent.
(
  export CUDA_VISIBLE_DEVICES=3
  export PYTHONUNBUFFERED=1
  echo "[Exp4] Starting on GPU 3 → $OUT4"
  python main.py $COMMON_ARGS \
      --out_dir "$OUT4" \
      --use_missing_gate \
      --paired_only \
      --pet_dropout_prob 0.3 \
      > "$OUT4/train.log" 2>&1
  echo "[Exp4] Done. Log: $OUT4/train.log"
) &
PID4=$!

# ── Wait for all 4 ────────────────────────────────────────────────────────────
echo ""
echo "All 4 experiments launched. Waiting..."
echo "  PID Exp1=$PID1  Exp2=$PID2  Exp3=$PID3  Exp4=$PID4"
echo ""

wait $PID1 && echo "[Exp1] ✓ finished" || echo "[Exp1] ✗ FAILED (exit $?)"
wait $PID2 && echo "[Exp2] ✓ finished" || echo "[Exp2] ✗ FAILED (exit $?)"
wait $PID3 && echo "[Exp3] ✓ finished" || echo "[Exp3] ✗ FAILED (exit $?)"
wait $PID4 && echo "[Exp4] ✓ finished" || echo "[Exp4] ✗ FAILED (exit $?)"

echo ""
echo "============================================================"
echo "  All experiments complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Exp1 (paired-only)      : $OUT1"
echo "  Exp2 (naive joint)      : $OUT2"
echo "  Exp3 (missing gate)     : $OUT3"
echo "  Exp4 (synthetic missing): $OUT4"
echo "============================================================"
