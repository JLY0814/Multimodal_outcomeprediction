"""
Global configuration for dual-branch 3D CNN experiment.
Spacing: 0.97 x 0.97 x 3.0 mm
Patch physical size (non-MIL): 120 x 120 x 120 mm
Patch voxel size   (non-MIL): 124 x 124 x 40
Patch physical size (MIL):    80 x 80 x 80 mm
Patch voxel size   (MIL):     82 x 82 x 27
"""

import os

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_ROOT = "/path/to/your/data"          # root with per-patient folders
CT_SUFFIX  = "_CT.nii.gz"                 # CT filename suffix
PET_SUFFIX = "_PET.nii.gz"               # PET filename suffix
MASK_SUFFIX = "_mask.nii.gz"             # ROI mask filename suffix
LABEL_CSV  = "/path/to/labels.csv"       # columns: patient_id, label (0/1)

# ── Spacing & patch ───────────────────────────────────────────────────────────
SPACING = (0.97, 0.97, 3.0)              # mm per voxel (x, y, z)
PATCH_MM = (120.0, 120.0, 120.0)        # physical patch size in mm (non-MIL)
# derived voxel counts ≈ (124, 124, 40)
PATCH_SIZE = tuple(
    int(round(pm / sp)) for pm, sp in zip(PATCH_MM, SPACING)
)                                        # (124, 124, 40)

# MIL uses a slightly smaller patch (fewer offset instances needed to cover tumour)
MIL_PATCH_MM   = (80.0, 80.0, 80.0)    # physical patch size in mm (MIL instances)
MIL_PATCH_SIZE = tuple(
    int(round(pm / sp)) for pm, sp in zip(MIL_PATCH_MM, SPACING)
)                                        # (82, 82, 27)

# ── Dual-scale branch (large patch, CT + Dose) ────────────────────────────────
USE_PET_ONLY            = False
USE_DUAL_SCALE          = False
LARGE_PATCH_MM          = (120.0, 120.0, 90.0)   # physical size of large patch in mm
LARGE_PATCH_SIZE        = tuple(
    int(round(pm / sp)) for pm, sp in zip(LARGE_PATCH_MM, SPACING)
)                                                  # => (124, 124, 30)
CT_LARGE_CHANNELS        = [8, 16, 32, 64, 128]  # lightweight backbone for CT_large
DOSE_LARGE_CHANNELS      = [8, 16, 32, 64, 128]  # lightweight backbone for Dose_large
DUAL_SCALE_FUSION_HIDDEN = 256                    # hidden dim in dual-scale fusion MLP
DUAL_SCALE_USE_CT_LARGE  = True                   # True  = Branch-2 uses CT_large + Dose_large
                                                  # False = Branch-2 uses Dose only

# ── Model ─────────────────────────────────────────────────────────────────────
DLA_CHANNELS  = [16, 32, 64, 128, 256]  # DLA stage output channels (CT & PET)
EMBED_DIM     = 256                      # after global avg pool per branch
FUSION_HIDDEN = 256                      # MLP hidden dim after concat
DROPOUT       = 0.5
NUM_CLASSES   = 1                        # sigmoid binary output

# ── Model size: 'small' variant (≈ 6-7 M params vs 26 M for 'base') ───────────
DLA_CHANNELS_SMALL  = [12, 24, 48, 96, 192]  # ~13M params (vs ~26M for base)
FUSION_HIDDEN_SMALL = 192

# ── Dose (third modality, optional) ───────────────────────────────────────────
USE_DOSE      = False                    # set True to enable dose branch
DOSE_CHANNELS = [8, 16, 32, 64, 128]    # shallower backbone for dose
DOSE_CHANNELS_SMALL = [6, 12, 24, 48, 96]
DOSE_MAX      = 70.0                    # clip Gy value → [0, 1] normalisation

# ── MIL-Attention (optional) ───────────────────────────────────────────────────
# Offsets (dx, dy, dz) in voxels relative to tumour centre.
# mil_patch_size=(82,82,27); shift ~25% in-plane → 21 vox, ~25% slice → 7 vox.
USE_MIL       = False
MIL_OFFSETS   = [                        # 7 instances by default
    ( 0,  0,  0),                        # 1. tumour centre
    (21,  0,  0),                        # 2. +x  (~20 mm)
    (-21, 0,  0),                        # 3. -x
    ( 0, 21,  0),                        # 4. +y
    ( 0,-21,  0),                        # 5. -y
    ( 0,  0,  7),                        # 6. +z  (~21 mm)
    ( 0,  0, -7),                        # 7. -z
]
MIL_ATTN_DIM  = 128                     # hidden dim of gated attention network
MIL_N_SLICES   = 5      # K: bg-patch mode: uniformly sample K slices from tumor z range
BG_MIN_DIST_MM = 50.0   # min physical distance (mm) from tumor mask for bg patch center

# ── Global CT+Dose branch ─────────────────────────────────────────────────────
# Full-volume CT (dose-modulated) processed by a lightweight global encoder.
# Volumes are resized to GLOBAL_SIZE before batching.
USE_GLOBAL_BRANCH   = False
GLOBAL_SIZE         = (128, 128, 48)  # (X, Y, Z) target voxel size after resize
GLOBAL_ENC_CHANNELS = [16, 32, 64, 128]   # GlobalCTDoseEncoder stage channels
GLOBAL_OUT_DIM      = 128             # global token dimension
DOSE_ALPHA_INIT     = 0.25            # initial alpha for DoseModulation
FUSION_GATE_HIDDEN  = 64             # hidden dim of GatedFusion MLP

# ── Missing modality gate ──────────────────────────────────────────────────────
# Unified CT+PET training that also accepts CT-only patients (PET missing).
# The gate learns to suppress the PET embedding when PET is absent.
USE_MISSING_GATE    = False  # enable missing-aware gate in DualBranch3DCNN
PET_DROPOUT_PROB    = 0.25   # prob of zeroing PET per paired sample during training
MISSING_GATE_HIDDEN = 64     # hidden dim of the gate MLP

# ── Training ──────────────────────────────────────────────────────────────────
SEED          = 42
NUM_FOLDS     = 5
EPOCHS        = 100
BATCH_SIZE    = 8                        # half pos / half neg per batch
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
LR_PATIENCE   = 15                       # ReduceLROnPlateau patience
EARLY_STOP    = 30                       # patience in epochs
MIN_EPOCHS    = 70                       # early stopping cannot trigger before this epoch

# ── CT-branch fine-tune phase (use_ct_finetune) ───────────────────────────────
CT_FINETUNE_EPOCHS = 30                  # fine-tune epochs after main training
CT_FINETUNE_LR     = 1e-5               # LR for CT branch fine-tuning
CT_AUX_WEIGHT  = 0.0   # auxiliary CT-only head loss weight (0 = disabled)
AMP           = True                     # mixed precision

# ── Augmentation ──────────────────────────────────────────────────────────────
GRIDMASK_RATIO   = 0.4                  # fraction of voxels masked per aug
GRIDMASK_PROB    = 0.5                  # probability of applying GridMask
MIXUP_ALPHA      = 0.4                  # Beta distribution alpha for MixUp
MIXUP_PROB       = 0.5                  # probability of applying MixUp
FLIP_PROB        = 0.5
ROTATE_PROB      = 0.3
ROTATE_RANGE     = 15                   # degrees

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
