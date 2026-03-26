#!/usr/bin/env bash
# run_patch_size_comparison.sh
# Compare MIL models trained with two different patch sizes.
#
# 用法:
#   bash run_patch_size_comparison.sh single   # 只比较 fold 1
#   bash run_patch_size_comparison.sh multi    # 全部 5 折 + 汇总（默认）

set -euo pipefail

# ── 数据路径 ──────────────────────────────────────────────────────────────────
CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"

# ── 两组模型目录 ──────────────────────────────────────────────────────────────
# 修改为实际的训练输出目录
DIR_A="/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/dual_branch_3d_cnn/outputs/SMALL_PATCH_DIR"
DIR_B="/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/dual_branch_3d_cnn/outputs/LARGE_PATCH_DIR"

# ── Patch 物理尺寸 (mm, x y z 三个值) ────────────────────────────────────────
# 必须与训练时 config.py 中 MIL_PATCH_MM 的实际值一致
PATCH_MM_A="80 80 80"     # Group A: 小 patch
PATCH_MM_B="120 120 120"  # Group B: 大 patch

# ── 显示标签 ──────────────────────────────────────────────────────────────────
LABEL_A="small_80mm"
LABEL_B="large_120mm"

# ── MIL offsets（留空则使用 config.py 中的 MIL_OFFSETS）─────────────────────
# 如果两组 patch size 对应不同的 offsets，在此填入展开的 dx dy dz 序列
# 例如: MIL_OFFSETS_A="0 0 0  21 0 0  -21 0 0  0 21 0  0 -21 0  0 0 7  0 0 -7"
# 留空 = 两组均使用 config.py 的默认 MIL_OFFSETS
MIL_OFFSETS_A=""
MIL_OFFSETS_B=""

# ── GPU 选择 ──────────────────────────────────────────────────────────────────
# 留空 = 使用系统默认；填卡号如 "2" 则指定
# 查看空闲卡: nvidia-smi --query-gpu=index,memory.free --format=csv
GPUS=""

# ── 其他设置 ──────────────────────────────────────────────────────────────────
MODE="${1:-multi}"        # single 或 multi
FOLD=1                    # single 模式使用的折数
VAL_ONLY="--val_only"     # 推荐保留，保证使用训练时相同的 val 集
MODEL_SIZE="base"         # base 或 small
INNER_FRAC=0.5            # 中心区域比例（center/periphery ablation）
AMP="--amp"
INFER_BS=64               # OOM 时调小
EXPORT_BS=4
NUM_WORKERS=4
N_VIS_CASES=4             # 可视化展示的病例数

OUT_DIR="./outputs/patch_size_comparison_$(date +%Y%m%d_%H%M%S)"

# ── 应用 GPU 选择 ─────────────────────────────────────────────────────────────
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# ── 组装 offsets 参数 ─────────────────────────────────────────────────────────
OFFSETS_FLAGS_A=""
if [ -n "$MIL_OFFSETS_A" ]; then
    OFFSETS_FLAGS_A="--mil_offsets_a $MIL_OFFSETS_A"
fi
OFFSETS_FLAGS_B=""
if [ -n "$MIL_OFFSETS_B" ]; then
    OFFSETS_FLAGS_B="--mil_offsets_b $MIL_OFFSETS_B"
fi

echo "============================================================"
echo "  Patch Size Comparison"
echo "  Mode    : $MODE"
echo "  Group A : $LABEL_A  [$DIR_A]"
echo "  Group B : $LABEL_B  [$DIR_B]"
echo "  Patch A : ${PATCH_MM_A} mm"
echo "  Patch B : ${PATCH_MM_B} mm"
echo "  Output  : $OUT_DIR"
echo "============================================================"

python patch_size_comparison.py "$MODE"     \
    --dir_a             "$DIR_A"            \
    --dir_b             "$DIR_B"            \
    --label_a           "$LABEL_A"          \
    --label_b           "$LABEL_B"          \
    --patch_mm_a        $PATCH_MM_A         \
    --patch_mm_b        $PATCH_MM_B         \
    --csv               "$CSV"              \
    --ct_dir            "$CT_DIR"           \
    --pet_dir           "$PET_DIR"          \
    --mask_dir          "$MASK_DIR"         \
    --label_col         recurrence          \
    --ct_axis_order     ZYX                 \
    --pet_axis_order    XYZ                 \
    --ct_wl             40                  \
    --ct_ww             400                 \
    --pet_max           200                 \
    --model_size        "$MODEL_SIZE"       \
    --out_dir           "$OUT_DIR"          \
    --fold              $FOLD               \
    --inner_frac        $INNER_FRAC         \
    --infer_batch_size  $INFER_BS           \
    --export_batch_size $EXPORT_BS          \
    --num_workers       $NUM_WORKERS        \
    --n_vis_cases       $N_VIS_CASES        \
    $VAL_ONLY $AMP                          \
    $OFFSETS_FLAGS_A $OFFSETS_FLAGS_B

echo ""
echo "============================================================"
echo "  完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: $OUT_DIR"
echo "============================================================"
