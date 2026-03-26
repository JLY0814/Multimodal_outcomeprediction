#!/usr/bin/env bash
# run_all.sh — Batch runner: 3 patch-size groups × 14 modes = 42 experiments
# All experiments run sequentially; logs saved to spring_output/<tag>/train.log
#
# Usage: bash run_all.sh [recurrence|figo]
#
# Modes per patch-size group:
#   1.  single patch  + missing gate,  dropout = 0.0
#   2.  single patch  + missing gate,  dropout = 0.1
#   3.  single patch  + missing gate,  dropout = 0.2
#   4.  single patch  + missing gate,  dropout = 0.3
#   5.  MIL           + missing gate,  dropout = 0.0
#   6.  MIL           + missing gate,  dropout = 0.1
#   7.  MIL           + missing gate,  dropout = 0.2
#   8.  MIL           + missing gate,  dropout = 0.3
#   9.  single patch  (no missing gate)
#   10. MIL           (no missing gate)
#   11. single patch  + missing gate + global dose,  dropout = 0.2
#   12. MIL           + missing gate + global dose,  dropout = 0.2
#   13. single patch  + CT-only
#   14. MIL           + CT-only

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"
DOSE_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"

# ── Fixed settings ─────────────────────────────────────────────────────────────
LABEL_COL="${1:-recurrence}"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2
export PYTHONUNBUFFERED=1
BASE_OUT_DIR="./spring_output"
mkdir -p "$BASE_OUT_DIR"

# ── Patch-size groups (XYZ voxels, spacing = 0.97 × 0.97 × 3.0 mm) ───────────
# Group index:         0            1            2
MIL_SIZES=(   "103 103 33" "113 113 37" "124 124 40" )   # 100 / 110 / 120 mm
SINGLE_SIZES=( "144 144 47" "155 155 50" "165 165 53" )  # 140 / 150 / 160 mm
SIZE_TAGS=(    "mil100"      "mil110"      "mil120"    )

# ── Common args shared by every experiment ─────────────────────────────────────
COMMON_ARGS=(
    --csv            "$CSV"
    --ct_dir         "$CT_DIR"
    --pet_dir        "$PET_DIR"
    --mask_dir       "$MASK_DIR"
    --label_col      "$LABEL_COL"
    --ct_axis_order  ZYX
    --pet_axis_order XYZ
    --ct_wl          40
    --ct_ww          400
    --pet_max        200
    --num_folds      5
    --seed           42
    --num_workers    4
    --preload
)

# ── Experiment runner ──────────────────────────────────────────────────────────
TOTAL=48
EXP_NUM=0

run_exp() {
    local tag="$1"; shift
    EXP_NUM=$(( EXP_NUM + 1 ))

    # ── Skip if a previous run with this tag already completed all 5 folds ──
    local existing
    existing=$(find "$BASE_OUT_DIR" -maxdepth 1 -type d -name "${tag}_*" 2>/dev/null \
               | sort | tail -1)
    if [ -n "$existing" ] && [ -f "${existing}/fold5_best.pth" ]; then
        echo ""
        echo "  [SKIP ${EXP_NUM}/${TOTAL}] ${tag} — already complete (${existing})"
        return 0
    fi

    local out_dir="${BASE_OUT_DIR}/${tag}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out_dir"
    local port
    port=$(shuf -i 29500-29999 -n 1)

    echo ""
    echo "============================================================"
    echo "  Experiment ${EXP_NUM}/${TOTAL} : ${tag}"
    echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Out dir : ${out_dir}"
    echo "============================================================"

    if torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${port}" main.py \
        "${COMMON_ARGS[@]}" \
        --out_dir "$out_dir" \
        "$@" \
        2>&1 | tee "${out_dir}/train.log"; then
        echo ""
        echo "  Finished : $(date '+%Y-%m-%d %H:%M:%S')  → ${out_dir}"
    else
        echo ""
        echo "  [FAILED ${EXP_NUM}/${TOTAL}] ${tag} — skipping to next experiment"
        echo "  Failed at : $(date '+%Y-%m-%d %H:%M:%S')  log: ${out_dir}/train.log"
    fi
}

# ── Main loop ──────────────────────────────────────────────────────────────────
for i in 0 1 2; do
    read -r MX MY MZ <<< "${MIL_SIZES[$i]}"
    read -r SX SY SZ <<< "${SINGLE_SIZES[$i]}"
    STAG="${SIZE_TAGS[$i]}"

    # ── Modes 1-4: single patch + missing gate, 4 dropout values ────────────
    for dp in 0.0 0.1 0.2 0.3; do
        dp_tag="${dp/./}"          # e.g. 0.1 → 01
        run_exp "single_mg_dp${dp_tag}_${STAG}" \
            --patch_size     $SX $SY $SZ \
            --mil_patch_size $MX $MY $MZ \
            --use_missing_gate \
            --paired_only_val \
            --pet_dropout_prob "$dp"
    done

    # ── Modes 5-8: MIL + missing gate, 4 dropout values ─────────────────────
    for dp in 0.0 0.1 0.2 0.3; do
        dp_tag="${dp/./}"
        run_exp "mil_mg_dp${dp_tag}_${STAG}" \
            --patch_size     $SX $SY $SZ \
            --mil_patch_size $MX $MY $MZ \
            --use_mil \
            --use_missing_gate \
            --paired_only_val \
            --pet_dropout_prob "$dp"
    done

    # ── Mode 9: single patch, no missing gate ─────────────────────────────────
    run_exp "single_standard_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ

    # ── Mode 10: MIL, no missing gate ────────────────────────────────────────
    run_exp "mil_standard_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_mil

    # ── Mode 11: single patch + missing gate + global dose, dropout=0.2 ──────
    run_exp "single_mg_global_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_missing_gate \
        --paired_only_val \
        --pet_dropout_prob 0.2 \
        --use_global_branch \
        --dose_dir "$DOSE_DIR" \
        --dose_axis_order ZYX

    # ── Mode 12: MIL + missing gate + global dose, dropout=0.2 ──────────────
    run_exp "mil_mg_global_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_mil \
        --use_missing_gate \
        --paired_only_val \
        --pet_dropout_prob 0.2 \
        --use_global_branch \
        --dose_dir "$DOSE_DIR" \
        --dose_axis_order ZYX

    # ── Mode 13: single patch, CT-only ───────────────────────────────────────
    run_exp "single_ctonly_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_ct_only

    # ── Mode 14: MIL, CT-only ────────────────────────────────────────────────
    run_exp "mil_ctonly_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_mil \
        --use_ct_only

    # ── Mode 15: single patch + missing gate + paired-only, dropout=0.0 ──────
    run_exp "single_mg_pairedonly_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_missing_gate \
        --paired_only \
        --pet_dropout_prob 0.0

    # ── Mode 16: MIL + missing gate + paired-only, dropout=0.0 ───────────────
    run_exp "mil_mg_pairedonly_${STAG}" \
        --patch_size     $SX $SY $SZ \
        --mil_patch_size $MX $MY $MZ \
        --use_mil \
        --use_missing_gate \
        --paired_only \
        --pet_dropout_prob 0.0

done

echo ""
echo "============================================================"
echo "  All ${TOTAL} experiments complete : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results in : ${BASE_OUT_DIR}"
echo "============================================================"
