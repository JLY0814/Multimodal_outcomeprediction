#!/usr/bin/env bash
# run_mil_analysis.sh — MIL mechanistic analysis
#
# 前提：checkpoint 必须来自用 --use_mil 训练的模型。
#
# 用法：
#   # 单模型
#   bash run_mil_analysis.sh single outputs/xxx/fold1_best.pth
#
#   # 多模型：逐个传路径
#   bash run_mil_analysis.sh multi  outputs/xxx/fold1_best.pth fold2_best.pth ...
#
#   # 多模型：直接传训练输出目录（自动发现 fold*_best.pth）
#   bash run_mil_analysis.sh multi  outputs/mil_xxx_recurrence_20260223/

set -euo pipefail

# ── 数据路径（与 run.sh 保持一致）──────────────────────────────────────────────
CSV="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"

# ── BG-patch 模式开关（分析用 --use_bg_patches 训练的模型时设为 true）────────────
# 必须与训练时 run.sh 中 BG_FLAGS 的参数一致！
USE_BG_PATCHES=true
N_SLICES=3           # 对应训练时的 --n_slices
BG_MIN_DIST_MM=50    # 对应训练时的 --bg_min_dist_mm

# ── 默认 single 模型路径（single 模式不传路径时使用）────────────────────────────
# 留空 ("") 则 single 模式仍必须在命令行传入 checkpoint 路径
DEFAULT_SINGLE_CHECKPOINT=""

# ── 默认多模型目录（multi 模式不传路径时使用）────────────────────────────────────
# 普通 MIL 模型（USE_BG_PATCHES=false）
DEFAULT_MIL_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/dual_branch_3d_cnn/outputs/MIL_120_ual_branch_recurrence_20260223_213324"
# BG-patch MIL 模型（USE_BG_PATCHES=true）
DEFAULT_BG_DIR="/shared/anastasio-s3/jyue/Large-Scale-Medical/Downstream/dual_branch_3d_cnn/outputs/olddata_BG_dual_branch_mil_recurrence_20260226_110954"

# ── GPU 选择 ─────────────────────────────────────────────────────────────────────
# 设为具体卡号（如 "2" 或 "0,1"）；留空则使用系统默认
# 查看空闲卡：nvidia-smi --query-gpu=index,memory.free --format=csv
GPUS=""

# ── 速度选项 ────────────────────────────────────────────────────────────────────
AMP="--amp"          # float16 推理；如 GPU 不支持可改为 ""
INFER_BS=128         # 每次 GPU 调用的最大变体数；OOM 时调小（如 32）
EXPORT_BS=4          # export_attention 的病人 batch size
NUM_WORKERS=4        # DataLoader worker 数

# ── 其他选项 ────────────────────────────────────────────────────────────────────
LABEL_COL="recurrence"   # 或 figo
MODEL_SIZE="base"         # 或 small
N_RAND=10                 # top-k removal 的随机试验次数
N_TOP_CASES=6             # 非均匀可视化展示的病例数（per-case PNG）
N_STD_GROUP=10            # high-std vs low-std 对比组每组病例数
NONUNIFORM_ONLY=false    # true → 只跑非均匀性分析，跳过 ring/topk/modality（速度更快）
NO_BASELINE_NORM=false   # true → 关闭 nonuniform 分析中的 per-patient 归一化，改用原始 log-odds drop
VAL_ONLY=""     # 只在各 fold 的 val 集上分析；改为 "" 则用全量数据
FOLD=1                    # single 模式下使用第几折的 val 集（1-5）

# ══════════════════════════════════════════════════════════════════════════════
# 内部：根据 USE_BG_PATCHES 组装附加 flags
# ══════════════════════════════════════════════════════════════════════════════
BG_FLAGS=""
if [ "$USE_BG_PATCHES" = "true" ]; then
    BG_FLAGS="--use_bg_patches --n_slices $N_SLICES --bg_min_dist_mm $BG_MIN_DIST_MM"
fi

NONUNIFORM_FLAG=""
if [ "$NONUNIFORM_ONLY" = "true" ]; then
    NONUNIFORM_FLAG="--nonuniform_only"
fi

NO_BASELINE_NORM_FLAG=""
if [ "$NO_BASELINE_NORM" = "true" ]; then
    NO_BASELINE_NORM_FLAG="--no_patient_baseline_norm"
fi

# 输出目录名体现分析模式
if [ "$USE_BG_PATCHES" = "true" ]; then
    _ANALYSIS_TAG="bg_mil_analysis"
else
    _ANALYSIS_TAG="mil_analysis"
fi

# ── 应用 GPU 选择 ────────────────────────────────────────────────────────────────
if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 解析模式参数
# ══════════════════════════════════════════════════════════════════════════════
MODE="${1:-multi}"
[ "$#" -gt 0 ] && shift

# ── 工具函数：从目录或文件列表中收集 checkpoint ─────────────────────────────────
collect_checkpoints() {
    # 接受若干参数：
    #   - 如果只有一个参数且是目录 → 在该目录下查找 fold*_best.pth
    #   - 否则 → 每个参数都当作 checkpoint 路径
    local -a result=()
    if [ "$#" -eq 1 ] && [ -d "$1" ]; then
        while IFS= read -r f; do
            result+=("$f")
        done < <(find "$1" -maxdepth 1 -name "fold*_best.pth" | sort)
        if [ "${#result[@]}" -eq 0 ]; then
            echo "错误：目录 $1 下未找到 fold*_best.pth 文件" >&2
            exit 1
        fi
    else
        result=("$@")
    fi
    printf '%s\n' "${result[@]}"
}

# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "single" ]; then
# ══════════════════════════════════════════════════════════════════════════════

    CHECKPOINT="${1:-$DEFAULT_SINGLE_CHECKPOINT}"
    if [ -z "$CHECKPOINT" ]; then
        echo "错误：请提供 checkpoint 路径或在脚本顶部设置 DEFAULT_SINGLE_CHECKPOINT" >&2
        echo "  例如：bash run_mil_analysis.sh single outputs/xxx/fold1_best.pth" >&2
        exit 1
    fi
    OUT_DIR="./outputs/${_ANALYSIS_TAG}_single_$(date +%Y%m%d_%H%M%S)"

    echo "============================================================"
    echo "  MIL Analysis [single-model]"
    echo "  Checkpoint   : $CHECKPOINT"
    echo "  BG-patch mode: $USE_BG_PATCHES"
    if [ "$USE_BG_PATCHES" = "true" ]; then
        echo "  n_slices     : $N_SLICES"
        echo "  bg_min_dist  : ${BG_MIN_DIST_MM} mm"
    fi
    echo "  Val-only fold: $FOLD"
    echo "  Output       : $OUT_DIR"
    echo "============================================================"

    python mil_analysis.py \
        --checkpoint        "$CHECKPOINT"   \
        --csv               "$CSV"          \
        --ct_dir            "$CT_DIR"       \
        --pet_dir           "$PET_DIR"      \
        --mask_dir          "$MASK_DIR"     \
        --label_col         "$LABEL_COL"    \
        --ct_axis_order     ZYX             \
        --pet_axis_order    XYZ             \
        --ct_wl             40              \
        --ct_ww             400             \
        --pet_max           200             \
        --model_size        "$MODEL_SIZE"   \
        --out_dir           "$OUT_DIR"      \
        --infer_batch_size  "$INFER_BS"     \
        --export_batch_size "$EXPORT_BS"    \
        --num_workers       "$NUM_WORKERS"  \
        --n_rand_trials     "$N_RAND"       \
        --n_top_cases       "$N_TOP_CASES"  \
        --n_std_group       "$N_STD_GROUP"  \
        --fold              "$FOLD"         \
        $VAL_ONLY \
        $BG_FLAGS \
        $NONUNIFORM_FLAG \
        $NO_BASELINE_NORM_FLAG \
        $AMP

# ══════════════════════════════════════════════════════════════════════════════
elif [ "$MODE" = "multi" ]; then
# ══════════════════════════════════════════════════════════════════════════════

    # 若没有额外参数，按 USE_BG_PATCHES 选择默认目录
    if [ "$#" -eq 0 ]; then
        if [ "$USE_BG_PATCHES" = "true" ]; then
            set -- "$DEFAULT_BG_DIR"
        else
            set -- "$DEFAULT_MIL_DIR"
        fi
    fi

    # 收集 checkpoint（支持传目录或逐个传文件）
    mapfile -t CHECKPOINTS < <(collect_checkpoints "$@")

    OUT_DIR="./outputs/${_ANALYSIS_TAG}_multi_$(date +%Y%m%d_%H%M%S)"

    echo "============================================================"
    echo "  MIL Analysis [multi-model: ${#CHECKPOINTS[@]} checkpoints]"
    for ck in "${CHECKPOINTS[@]}"; do echo "    $ck"; done
    echo "  BG-patch mode: $USE_BG_PATCHES"
    if [ "$USE_BG_PATCHES" = "true" ]; then
        echo "  n_slices     : $N_SLICES"
        echo "  bg_min_dist  : ${BG_MIN_DIST_MM} mm"
    fi
    echo "  Output       : $OUT_DIR"
    echo "============================================================"

    python mil_analysis.py \
        --checkpoints       "${CHECKPOINTS[@]}" \
        --csv               "$CSV"          \
        --ct_dir            "$CT_DIR"       \
        --pet_dir           "$PET_DIR"      \
        --mask_dir          "$MASK_DIR"     \
        --label_col         "$LABEL_COL"    \
        --ct_axis_order     ZYX             \
        --pet_axis_order    XYZ             \
        --ct_wl             40              \
        --ct_ww             400             \
        --pet_max           200             \
        --model_size        "$MODEL_SIZE"   \
        --out_dir           "$OUT_DIR"      \
        --infer_batch_size  "$INFER_BS"     \
        --export_batch_size "$EXPORT_BS"    \
        --num_workers       "$NUM_WORKERS"  \
        --n_rand_trials     "$N_RAND"       \
        --n_top_cases       "$N_TOP_CASES"  \
        --n_std_group       "$N_STD_GROUP"  \
        $VAL_ONLY \
        $BG_FLAGS \
        $NONUNIFORM_FLAG \
        $NO_BASELINE_NORM_FLAG \
        $AMP

# ══════════════════════════════════════════════════════════════════════════════
else
    echo "错误：第一个参数必须是 single 或 multi，收到：$MODE"
    exit 1
fi

echo ""
echo "============================================================"
echo "  完成：$(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果：$OUT_DIR"
echo "============================================================"
