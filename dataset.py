"""
Dataset for dual-branch 3-D CNN.

Files are stored as .npy arrays named:
    {ct_dir}/{Patient_ID}_CT.npy
    {pet_dir}/{Patient_ID}_PET.npy
    {mask_dir}/{Patient_ID}_MTV_Cervix.npy   (case-insensitive, preferred)
    {mask_dir}/{Patient_ID}_Cervix_new.npy   (case-insensitive, fallback)

Mask is used to compute the tumour centroid (ROI centre).
Falls back to volume centre if no mask is found for a patient.

Supported label columns (mirrors CT_CNN_Baseline logic):
  label_col='recurrence'  → CSV column 'Recurrence'  (Yes/No or 1/0)
  label_col='figo'        → CSV column 'FIGO 2018 Stage'
                            early  (I / II)   → 0
                            advanced (III/IV) → 1
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from config import (PATCH_SIZE, MIL_PATCH_SIZE, LARGE_PATCH_SIZE,
                    MIL_OFFSETS as _DEFAULT_MIL_OFFSETS,
                    GRIDMASK_PROB, GRIDMASK_RATIO, FLIP_PROB, ROTATE_PROB,
                    SPACING, GLOBAL_SIZE)
from utils import extract_patch, normalise_ct, normalise_pet, normalise_dose, get_roi_center
from augmentation import Augment3D, mixup_3d


# ── Label parsing (same logic as CT_CNN_Baseline) ────────────────────────────

FIGO_EARLY    = {'IA1', 'IA2', 'IB1', 'IB2', 'IB3', 'IIA1', 'IIA2', 'IIB'}
FIGO_ADVANCED = {'IIIA', 'IIIB', 'IIIC1', 'IIIC2', 'IVA', 'IVB'}


def parse_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Add a 'label' column (int 0/1) to df; drop rows with unknown values."""
    if label_col == 'recurrence':
        required = {'Patient_ID', 'Recurrence'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.dropna(subset=['Patient_ID', 'Recurrence']).copy()
        rec = df['Recurrence']
        if rec.dtype == object:
            rec = rec.str.strip().str.lower().map({'yes': 1, 'no': 0})
            if rec.isna().any():
                raise ValueError("Recurrence column has unexpected values (expected Yes/No or 0/1)")
        df['label'] = rec.astype(int)

    elif label_col == 'figo':
        required = {'Patient_ID', 'FIGO 2018 Stage'}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")
        df = df.dropna(subset=['Patient_ID', 'FIGO 2018 Stage']).copy()
        df['FIGO 2018 Stage'] = df['FIGO 2018 Stage'].str.strip()
        valid = df['FIGO 2018 Stage'].isin(FIGO_EARLY | FIGO_ADVANCED)
        n_drop = (~valid).sum()
        if n_drop:
            print(f"[Dataset] Dropped {n_drop} rows with unrecognised FIGO stage.")
        df = df[valid].copy()
        df['label'] = df['FIGO 2018 Stage'].map(
            {s: 0 for s in FIGO_EARLY} | {s: 1 for s in FIGO_ADVANCED}
        ).astype(int)

    else:
        raise ValueError(f"label_col must be 'figo' or 'recurrence', got '{label_col}'")

    return df.reset_index(drop=True)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_npy(path: str, axis_order: str = 'ZYX') -> np.ndarray:
    """
    Load a .npy file and return a (X, Y, Z) float32 array.

    axis_order='ZYX' → file stored as (Z, Y, X), transpose to (X, Y, Z).
                       Matches baseline ct_axis_order='ZYX' behaviour.
    axis_order='XYZ' → file already in (X, Y, Z) order, no transpose.
                       Use this for PET when stored in "normal" order.
    """
    arr = np.load(path).astype(np.float32)
    if axis_order.upper() == 'ZYX':
        arr = np.transpose(arr, (2, 1, 0))  # (Z,Y,X) → (X,Y,Z)
    # XYZ: already correct, no transpose
    return arr


def volume_center(vol: np.ndarray) -> Tuple[int, int, int]:
    """Return the geometric centre of the volume (fallback when no mask)."""
    return tuple(s // 2 for s in vol.shape)


def build_mask_index(mask_dir: str) -> Dict[str, str]:
    """
    Scan mask_dir for cervix tumour mask .npy files.
    Returns { patient_id_lower → best_file_path }.

    Two accepted filename patterns (case-insensitive):
      A. Contains both 'mtv' and 'cervix'  e.g. {pid}_MTV_Cervix.npy
      B. Contains 'cervix' only            e.g. {pid}_Cervix_new.npy

    patient_id = everything in the filename before the earliest of
                 'mtv' / 'cervix' found.

    Selection priority (lower number = higher priority):
      1. Pattern A, standard  — no 'old' or 'opt'   (best)
      2. Pattern A, OPT       — contains 'opt'
      3. Pattern A, OLD       — contains 'old'
      4. Pattern B, standard  — no 'old' or 'opt'
      5. Pattern B, OPT       — contains 'opt'
      6. Pattern B, OLD       — contains 'old'       (worst)
    Within the same priority tier the alphabetically first path wins.
    """
    if not mask_dir:
        return {}

    def _file_priority(fname_lower: str) -> int:
        has_mtv = "mtv" in fname_lower
        base = 1 if has_mtv else 4          # pattern A beats pattern B
        if "old" in fname_lower:
            return base + 2
        if "opt" in fname_lower:
            return base + 1
        return base

    # pid_lower → (priority, path)
    candidates: Dict[str, tuple] = {}

    for path in glob.glob(os.path.join(mask_dir, "*.npy")):
        fname = os.path.basename(path)
        fl    = fname.lower()
        if "cervix" not in fl:
            continue
        # Find cut point: earliest of 'mtv' or 'cervix'
        cut = fl.index("cervix")
        if "mtv" in fl:
            cut = min(cut, fl.index("mtv"))
        pid = fname[:cut].rstrip("_- ").lower()
        if not pid:
            continue
        priority = _file_priority(fl)
        existing = candidates.get(pid)
        if existing is None or priority < existing[0] or (
                priority == existing[0] and path < existing[1]):
            candidates[pid] = (priority, path)

    return {pid: info[1] for pid, info in candidates.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Dual-modality (CT + PET) patch dataset.

    Args:
        csv_path:       path to CSV (Patient_ID + label column).
        ct_dir:         directory with {Patient_ID}_CT.npy files.
        pet_dir:        directory with {Patient_ID}_PET.npy files.
        mask_dir:       directory with {Patient_ID}_MTV_Cervix.npy masks
                        (case-insensitive). None → fallback to volume centre.
        label_col:      'recurrence' or 'figo'.
        patch_size:     (D, H, W) voxel extent of output patch.
        ct_axis_order:  axis order of CT .npy ('ZYX' → transpose, 'XYZ' → keep).
        pet_axis_order: axis order of PET .npy ('ZYX' → transpose, 'XYZ' → keep).
        augment:        apply spatial / intensity augmentations.
        mixup_prob:     probability of MixUp per sample.
        mixup_alpha:    Beta distribution α for MixUp.
        ct_wl/ct_ww:    CT window level / width (HU).
        pet_max:        PET SUV clip maximum for [0,1] normalisation.
        use_dose:       if True, load Dose and return it as third modality.
        dose_dir:       directory with {Patient_ID}_Dose.npy files.
        dose_axis_order:axis order of Dose .npy (default same as CT: 'ZYX').
        dose_max:       Dose clip maximum in Gy for [0,1] normalisation.
        use_mil:        if True, return N instances per patient instead of one.
        mil_offsets:    list of (dx, dy, dz) voxel offsets from tumour centre.
                        Each offset → one instance patch. Defaults to 7 offsets.
        patient_ids:    if not None, restrict to this list of IDs.
        preload:        load all volumes into RAM at init.
    """

    def __init__(
        self,
        csv_path:        str,
        ct_dir:          str,
        pet_dir:         str,
        mask_dir:        Optional[str] = None,
        label_col:       str   = 'recurrence',
        patch_size:      Tuple = PATCH_SIZE,
        ct_axis_order:   str   = 'ZYX',
        pet_axis_order:  str   = 'XYZ',
        augment:         bool  = False,
        mixup_prob:      float = 0.0,
        mixup_alpha:     float = 0.4,
        ct_wl:           float = 40.0,
        ct_ww:           float = 400.0,
        pet_max:         float = 10.0,
        use_dose:        bool  = False,
        dose_dir:        Optional[str] = None,
        dose_axis_order: str   = 'ZYX',
        dose_max:        float = 70.0,
        use_mil:         bool  = False,
        mil_patch_size:  Tuple = MIL_PATCH_SIZE,
        mil_offsets:     Optional[List[tuple]] = None,
        patient_ids:     Optional[List[str]] = None,
        preload:         bool  = False,
        use_dual_scale:  bool  = False,
        large_patch_size: Tuple = LARGE_PATCH_SIZE,
        use_ct_only:     bool  = False,
        use_pet_only:    bool  = False,
        use_bg_patches:  bool  = False,   # enable slice-level tumor+bg paired sampling
        n_slices:        int   = 5,       # K: number of slices per patient → 2K patches
        bg_min_dist_mm:  float = 50.0,    # min distance (mm) from tumor mask for bg center
        use_missing_gate: bool = False,   # unified CT+PET loop; CT-only patients allowed
        paired_only:      bool = False,   # with use_missing_gate: exclude CT-only patients
        pet_dropout_prob: float = 0.0,    # prob of zeroing PET per sample during training
        use_global_branch: bool = False,  # also return full-volume CT+Dose for global branch
        global_size: Tuple    = GLOBAL_SIZE,  # (X, Y, Z) resize target for global volumes
        use_unipair:      bool = False,   # treat CT and PET as independent patients
        use_naive_joint:  bool = False,   # CT-only patients train with zero PET, single shared head
    ):
        self.ct_dir          = ct_dir
        self.pet_dir         = pet_dir
        self.patch_size      = patch_size
        self.ct_axis_order   = ct_axis_order
        self.pet_axis_order  = pet_axis_order
        self.augment         = augment
        self.mixup_prob      = mixup_prob
        self.mixup_alpha     = mixup_alpha
        self.ct_wl           = ct_wl
        self.ct_ww           = ct_ww
        self.pet_max         = pet_max
        self.use_dose        = use_dose
        self.dose_dir        = dose_dir
        self.dose_axis_order = dose_axis_order
        self.dose_max        = dose_max
        self.use_mil         = use_mil
        self.mil_patch_size  = mil_patch_size
        # Default: 7 instances (centre + 6 cardinal directions) — from config.MIL_OFFSETS
        if mil_offsets is None:
            mil_offsets = _DEFAULT_MIL_OFFSETS
        self.mil_offsets     = mil_offsets
        self.use_dual_scale  = use_dual_scale
        self.large_patch_size = large_patch_size
        self.use_ct_only      = use_ct_only
        self.use_pet_only     = use_pet_only
        self.use_bg_patches   = use_bg_patches
        self.n_slices         = n_slices
        self.bg_min_dist_px   = bg_min_dist_mm / SPACING[0]  # e.g. 50mm / 0.97mm ≈ 51.5 px
        self.use_missing_gate  = use_missing_gate
        self.paired_only       = paired_only
        self.pet_dropout_prob  = pet_dropout_prob
        self.use_global_branch = use_global_branch
        self.global_size       = global_size
        self.use_unipair       = use_unipair
        self.use_naive_joint   = use_naive_joint
        self.augmentor       = Augment3D(
            gridmask_prob  = GRIDMASK_PROB,
            gridmask_ratio = GRIDMASK_RATIO,
            flip_prob      = FLIP_PROB,
            rotate_prob    = ROTATE_PROB,
        ) if augment else None

        # Build case-insensitive mask lookup: pid_lower → file path
        self._mask_index = build_mask_index(mask_dir) if mask_dir else {}
        n_masks = len(self._mask_index)
        print(f"[Dataset] Mask index built: {n_masks} masks found"
              + (f" in {mask_dir}" if mask_dir else " (no mask_dir given, using volume centre)"))

        # ── Parse CSV + labels ────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        df = parse_labels(df, label_col)
        df['Patient_ID'] = df['Patient_ID'].astype(str)

        # Restrict to supplied patient_ids (for cross-val splits)
        if patient_ids is not None:
            patient_ids = [str(p) for p in patient_ids]
            df = df[df['Patient_ID'].isin(patient_ids)]

        # Filter missing files
        def _has_files(pid: str) -> bool:
            # Require mask for all patients when mask_dir is provided,
            # so every patch is centred on the tumour centroid.
            if mask_dir and not use_pet_only and pid.lower() not in self._mask_index:
                return False
            if use_unipair:
                # CT-only patients are allowed: they contribute only a CT sample (modality=0).
                # Paired patients contribute two samples (CT + PET).
                return os.path.exists(os.path.join(ct_dir, f"{pid}_CT.npy"))
            if use_naive_joint:
                # CT-only patients train with zero PET through the standard classifier.
                return os.path.exists(os.path.join(ct_dir, f"{pid}_CT.npy"))
            if use_pet_only:
                return os.path.exists(os.path.join(pet_dir, f"{pid}_PET.npy"))
            ct_ok = os.path.exists(os.path.join(ct_dir, f"{pid}_CT.npy"))
            if use_ct_only or (use_missing_gate and not paired_only):
                # Missing-gate mode: CT-only patients allowed (PET optional)
                return ct_ok
            pet_ok = os.path.exists(os.path.join(pet_dir, f"{pid}_PET.npy"))
            return ct_ok and pet_ok

        before = len(df)
        df = df[df['Patient_ID'].apply(_has_files)].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            print(f"[Dataset] Skipped {skipped} patients with missing CT/PET/mask files.")

        self.df     = df
        self.labels = df['label'].tolist()
        print(f"[Dataset] {len(df)} patients  |  "
              f"pos={sum(self.labels)}  neg={len(self.labels)-sum(self.labels)}")

        # Unipair index: maps virtual sample index → (patient_idx, modality).
        # Paired patients contribute 2 samples (modality 0=CT, 1=PET).
        # CT-only patients contribute 1 sample (modality 0 only).
        if use_unipair:
            self._unipair_is_paired = [
                os.path.exists(os.path.join(pet_dir, f"{str(pid)}_PET.npy"))
                for pid in self.df['Patient_ID'].astype(str)
            ]
            self._unipair_index = []
            for i, is_paired in enumerate(self._unipair_is_paired):
                self._unipair_index.append((i, 0))   # CT sample always
                if is_paired:
                    self._unipair_index.append((i, 1))  # PET sample only when paired
            n_paired = sum(self._unipair_is_paired)
            n_ctonly = len(self._unipair_is_paired) - n_paired
            print(f"[Dataset] unipair: {n_paired} paired ({2*n_paired} samples) + "
                  f"{n_ctonly} CT-only ({n_ctonly} samples) = "
                  f"{len(self._unipair_index)} total virtual samples")

        # Pre-build pos/neg index pools for MixUp partner selection
        self.pos_idx = [i for i, l in enumerate(self.labels) if l == 1]
        self.neg_idx = [i for i, l in enumerate(self.labels) if l == 0]

        # Optional preload
        self._cache: dict = {}
        self._mask_cache: dict = {}
        self._pet_present_cache: dict = {}
        if preload:
            print("Preloading volumes …")
            for _, row in df.iterrows():
                pid = row['Patient_ID']
                self._cache[pid] = self._load_raw(pid)
                self._mask_cache[pid] = self._load_mask(pid)
                if use_missing_gate:
                    pet_path = os.path.join(pet_dir, f"{pid}_PET.npy")
                    self._pet_present_cache[pid] = int(os.path.exists(pet_path))
            print("Preload done.")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_mask(self, pid: str) -> Optional[np.ndarray]:
        """
        Load the tumour mask for `pid` (case-insensitive lookup).
        Returns a binary float32 array in the same axis order as CT (ZYX→XYZ),
        or None if no mask file exists for this patient.
        """
        if pid in self._mask_cache:
            return self._mask_cache[pid]
        path = self._mask_index.get(pid.lower())
        if path is None:
            return None
        # Mask has the same axis order as PET (no transpose needed)
        mask = load_npy(path, self.pet_axis_order)
        return (mask > 0).astype(np.float32)

    def _load_dose(self, pid: str) -> Optional[np.ndarray]:
        """Load and normalise the dose volume; returns None if file missing."""
        if not self.use_dose or not self.dose_dir:
            return None
        path = os.path.join(self.dose_dir, f"{pid}_Dose.npy")
        if not os.path.exists(path):
            return None
        dose = load_npy(path, self.dose_axis_order)
        return normalise_dose(dose, max_gy=self.dose_max)

    def _load_raw(self, pid: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.use_pet_only:
            pet_path = os.path.join(self.pet_dir, f"{pid}_PET.npy")
            pet = load_npy(pet_path, self.pet_axis_order)
            return None, pet
        ct_path = os.path.join(self.ct_dir, f"{pid}_CT.npy")
        ct = load_npy(ct_path, self.ct_axis_order)
        if self.use_ct_only:
            return ct, None
        pet_path = os.path.join(self.pet_dir, f"{pid}_PET.npy")
        if not os.path.exists(pet_path):
            return ct, None  # CT-only patient in missing-gate mode
        pet = load_npy(pet_path, self.pet_axis_order)
        return ct, pet

    def _get_raw(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        if pid in self._cache:
            return self._cache[pid]
        return self._load_raw(pid)

    def _load_global(self, pid: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return full-volume CT and Dose tensors for the global branch, both
        resized to self.global_size and shaped (1, X, Y, Z).

        Uses _get_raw() so the CT benefits from the preload cache.
        Dose is loaded fresh each call (typically not cached).
        If no dose file exists, dose_global is all-zeros.
        """
        from scipy.ndimage import zoom

        ct_vol, _ = self._get_raw(pid)                                      # (X, Y, Z)
        ct_norm   = normalise_ct(ct_vol, wl=self.ct_wl, ww=self.ct_ww)     # [0, 1]

        # Resize CT to global_size
        factors_ct = tuple(g / s for g, s in zip(self.global_size, ct_norm.shape))
        ct_g = zoom(ct_norm, factors_ct, order=1).astype(np.float32)       # (gX, gY, gZ)

        # Dose: normalise then resize (zeros if missing)
        dose_vol = self._load_dose(pid)
        if dose_vol is not None:
            factors_dose = tuple(g / s for g, s in zip(self.global_size, dose_vol.shape))
            dose_g = zoom(dose_vol, factors_dose, order=1).astype(np.float32)
        else:
            dose_g = np.zeros(self.global_size, dtype=np.float32)

        return (
            torch.from_numpy(ct_g[np.newaxis]),      # (1, gX, gY, gZ)
            torch.from_numpy(dose_g[np.newaxis]),    # (1, gX, gY, gZ)
        )

    def _make_patch(self, pid: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (ct_patch, pet_patch, dose_patch) all shape (1, D, H, W).
        When use_ct_only=True,  pet_patch and dose_patch are None.
        When use_pet_only=True, ct_patch  and dose_patch are None.
        dose_patch is a zero array when use_dose=False or file is missing.
        """
        ct_vol, pet_vol = self._get_raw(pid)

        if self.use_pet_only:
            mask = self._load_mask(pid)
            center = get_roi_center(mask) if mask is not None else volume_center(pet_vol)
            pet_patch = normalise_pet(
                extract_patch(pet_vol, center, self.patch_size),
                max_suv=self.pet_max)
            return None, pet_patch[np.newaxis], None

        # Tumour centre: use ROI mask centroid if available, else volume centre
        mask = self._load_mask(pid)
        if mask is not None:
            center = get_roi_center(mask)
        else:
            center = volume_center(ct_vol)

        ct_patch = normalise_ct(
            extract_patch(ct_vol, center, self.patch_size),
            wl=self.ct_wl, ww=self.ct_ww)

        if self.use_ct_only:
            return ct_patch[np.newaxis], None, None

        if pet_vol is None:
            # CT-only patient in missing-gate mode: return zero PET
            pet_patch = np.zeros(self.patch_size, dtype=np.float32)
        else:
            pet_patch = normalise_pet(
                extract_patch(pet_vol, center, self.patch_size),
                max_suv=self.pet_max)

        # Dose patch
        dose_vol = self._load_dose(pid)
        if dose_vol is not None:
            dose_patch = extract_patch(dose_vol, center, self.patch_size)
        else:
            dose_patch = np.zeros(self.patch_size, dtype=np.float32)

        return (ct_patch[np.newaxis],      # (1, D, H, W)
                pet_patch[np.newaxis],
                dose_patch[np.newaxis])

    def _make_patches(self, pid: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        MIL mode: extract one patch per offset around the tumour centre.
        Returns stacked arrays of shape (N, 1, D, H, W) for CT, PET, Dose.
        When use_ct_only=True,  PET and Dose arrays are None.
        When use_pet_only=True, CT  and Dose arrays are None.
        """
        ct_vol, pet_vol = self._get_raw(pid)
        mask = self._load_mask(pid)

        if self.use_pet_only:
            base_center = get_roi_center(mask) if mask is not None else volume_center(pet_vol)
            pet_list = []
            for dx, dy, dz in self.mil_offsets:
                center = (base_center[0] + dx, base_center[1] + dy, base_center[2] + dz)
                pet_p = normalise_pet(
                    extract_patch(pet_vol, center, self.mil_patch_size),
                    max_suv=self.pet_max)
                pet_list.append(pet_p[np.newaxis])
            return None, np.stack(pet_list, axis=0), None

        base_center = get_roi_center(mask) if mask is not None else volume_center(ct_vol)
        dose_vol = self._load_dose(pid) if not self.use_ct_only else None

        ct_list, pet_list, dose_list = [], [], []
        for dx, dy, dz in self.mil_offsets:
            center = (base_center[0] + dx,
                      base_center[1] + dy,
                      base_center[2] + dz)
            ct_p = normalise_ct(
                extract_patch(ct_vol, center, self.mil_patch_size),
                wl=self.ct_wl, ww=self.ct_ww)
            ct_list.append(ct_p[np.newaxis])      # (1, D, H, W)

            if not self.use_ct_only:
                if pet_vol is None:
                    pet_p = np.zeros(self.mil_patch_size, dtype=np.float32)
                else:
                    pet_p = normalise_pet(
                        extract_patch(pet_vol, center, self.mil_patch_size),
                        max_suv=self.pet_max)
                if dose_vol is not None:
                    dose_p = extract_patch(dose_vol, center, self.mil_patch_size)
                else:
                    dose_p = np.zeros(self.mil_patch_size, dtype=np.float32)
                pet_list.append(pet_p[np.newaxis])
                dose_list.append(dose_p[np.newaxis])

        # Stack → (N, 1, D, H, W)
        ct_arr  = np.stack(ct_list, axis=0)
        if self.use_ct_only:
            return ct_arr, None, None
        return (ct_arr,
                np.stack(pet_list,  axis=0),
                np.stack(dose_list, axis=0))

    def _sample_bg_center_2d(self, mask_2d, ct_slice_2d, min_dist_px, rng):
        """
        Sample a background center (x, y) in a 2D slice satisfying:
        - distance from tumor mask > min_dist_px pixels
        - inside body region (ct > 0.01, proxy for non-air after normalisation)
        - at least patch_half away from image border (patch won't exceed bounds)
        Three-level fallback: full constraints → distance only → argmax distance.
        """
        from scipy.ndimage import distance_transform_edt
        dist_map = distance_transform_edt(~mask_2d.astype(bool))
        body     = ct_slice_2d > 0.01
        px_h = self.mil_patch_size[0] // 2
        py_h = self.mil_patch_size[1] // 2
        X, Y = mask_2d.shape
        border = np.zeros_like(mask_2d, dtype=bool)
        border[px_h:X - px_h, py_h:Y - py_h] = True

        valid = (dist_map > min_dist_px) & body & border
        pos   = np.argwhere(valid)
        if len(pos):
            return pos[rng.integers(len(pos))]

        valid2 = (dist_map > min_dist_px) & border
        pos2   = np.argwhere(valid2)
        if len(pos2):
            return pos2[rng.integers(len(pos2))]

        return np.array(np.unravel_index(dist_map.argmax(), dist_map.shape))

    def _make_patches_slice_bg(self, pid: str, rng=None):
        """
        Slice-level bag: uniformly sample K slices from the tumor z range;
        for each slice extract 1 tumor patch + 1 background patch → 2K patches total.

        Returns (ct, pet, dose, patch_types):
            ct, pet, dose : (2K, 1, D, H, W) arrays
            patch_types   : (2K,) int64 array, 0=tumor / 1=background

        Falls back to _make_patches() (all tumor, patch_types all 0) when mask is
        missing or empty.
        """
        if rng is None:
            rng = np.random.default_rng()
        ct_vol, pet_vol = self._get_raw(pid)
        mask             = self._load_mask(pid)
        dose_vol         = self._load_dose(pid) if not self.use_ct_only else None

        # Fallback: no mask → repeat offset-based patches to match 2*n_slices
        if mask is None or mask.sum() == 0:
            ct_a, pet_a, dose_a = self._make_patches(pid)
            target = 2 * self.n_slices
            n = ct_a.shape[0]
            if n != target:
                reps = (target + n - 1) // n
                ct_a = np.tile(ct_a, (reps, 1, 1, 1, 1))[:target]
                if pet_a is not None:
                    pet_a = np.tile(pet_a, (reps, 1, 1, 1, 1))[:target]
                if dose_a is not None:
                    dose_a = np.tile(dose_a, (reps, 1, 1, 1, 1))[:target]
            return ct_a, pet_a, dose_a, np.zeros(target, dtype=np.int64)

        # Identify tumor-containing slices (mask shape: X, Y, Z)
        z_has_tumor = mask.any(axis=(0, 1))           # (Z,) bool
        z_idx       = np.where(z_has_tumor)[0]
        z_min, z_max = int(z_idx[0]), int(z_idx[-1])

        K = self.n_slices
        if len(z_idx) >= K:
            targets    = np.linspace(z_min, z_max, K)
            selected_z = [z_idx[np.abs(z_idx - t).argmin()] for t in targets]
        else:
            selected_z = list(z_idx) + [z_idx[-1]] * (K - len(z_idx))

        ct_list, pet_list, dose_list, type_list = [], [], [], []

        for z_i in selected_z:
            mask_2d   = mask[:, :, z_i]              # (X, Y)
            positions = np.argwhere(mask_2d)
            if len(positions) == 0:
                positions = np.array([[mask.shape[0] // 2, mask.shape[1] // 2]])
            cxy = positions.mean(axis=0)              # (x, y) float

            # ── Tumor patch ──
            tc   = (int(round(float(cxy[0]))), int(round(float(cxy[1]))), int(z_i))
            ct_p = normalise_ct(extract_patch(ct_vol, tc, self.mil_patch_size),
                                wl=self.ct_wl, ww=self.ct_ww)
            ct_list.append(ct_p[np.newaxis])
            type_list.append(0)
            if not self.use_ct_only:
                if pet_vol is None:
                    pet_list.append(np.zeros((1, *self.mil_patch_size), dtype=np.float32))
                else:
                    pet_list.append(normalise_pet(
                        extract_patch(pet_vol, tc, self.mil_patch_size),
                        max_suv=self.pet_max)[np.newaxis])
                dose_p = (extract_patch(dose_vol, tc, self.mil_patch_size)
                          if dose_vol is not None
                          else np.zeros(self.mil_patch_size, dtype=np.float32))
                dose_list.append(dose_p[np.newaxis])

            # ── Background patch ──
            ct_slice_norm = normalise_ct(ct_vol[:, :, int(z_i)], wl=self.ct_wl, ww=self.ct_ww)
            bxy = self._sample_bg_center_2d(mask_2d, ct_slice_norm,
                                            self.bg_min_dist_px, rng)
            bc   = (int(bxy[0]), int(bxy[1]), int(z_i))
            ct_b = normalise_ct(extract_patch(ct_vol, bc, self.mil_patch_size),
                                wl=self.ct_wl, ww=self.ct_ww)
            ct_list.append(ct_b[np.newaxis])
            type_list.append(1)
            if not self.use_ct_only:
                if pet_vol is None:
                    pet_list.append(np.zeros((1, *self.mil_patch_size), dtype=np.float32))
                else:
                    pet_list.append(normalise_pet(
                        extract_patch(pet_vol, bc, self.mil_patch_size),
                        max_suv=self.pet_max)[np.newaxis])
                dose_b = (extract_patch(dose_vol, bc, self.mil_patch_size)
                          if dose_vol is not None
                          else np.zeros(self.mil_patch_size, dtype=np.float32))
                dose_list.append(dose_b[np.newaxis])

        ct_arr = np.stack(ct_list, axis=0)            # (2K, 1, D, H, W)
        types  = np.array(type_list, dtype=np.int64)  # (2K,)
        if self.use_ct_only:
            return ct_arr, None, None, types
        return (ct_arr,
                np.stack(pet_list,  axis=0),
                np.stack(dose_list, axis=0),
                types)

    def _make_large_patch(self, pid: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract large-patch CT and Dose centered on the tumour.

        Returns (ct_large, dose_large) each of shape (1, D_l, H_l, W_l).
        dose_large is zeros if dose_dir is not set or file is missing.
        """
        ct_vol, _ = self._get_raw(pid)
        mask = self._load_mask(pid)
        center = get_roi_center(mask) if mask is not None else volume_center(ct_vol)

        ct_large = normalise_ct(
            extract_patch(ct_vol, center, self.large_patch_size),
            wl=self.ct_wl, ww=self.ct_ww)

        # Load dose for large patch independently of use_dose flag for small branch
        dose_large_arr = None
        if self.dose_dir:
            path = os.path.join(self.dose_dir, f"{pid}_Dose.npy")
            if os.path.exists(path):
                dose_vol = load_npy(path, self.dose_axis_order)
                from utils import normalise_dose as _nd
                dose_vol = _nd(dose_vol, max_gy=self.dose_max)
                dose_large_arr = extract_patch(dose_vol, center, self.large_patch_size)
        if dose_large_arr is None:
            dose_large_arr = np.zeros(self.large_patch_size, dtype=np.float32)

        return (ct_large[np.newaxis], dose_large_arr[np.newaxis])  # (1, D_l, H_l, W_l)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.use_unipair:
            return len(self._unipair_index)
        return len(self.df)

    def __getitem__(self, idx: int):
        # In unipair mode: each physical patient maps to two virtual indices.
        # Remap idx → (patient_idx, modality) before retry logic so retries
        # pick a different *patient*, not a different virtual index.
        if self.use_unipair:
            patient_idx, modality = self._unipair_index[idx]
            n_virtual = len(self._unipair_index)
            for _attempt in range(10):
                try:
                    return self._getitem_unipair(patient_idx, modality)
                except Exception as e:
                    import warnings
                    row = self.df.iloc[patient_idx]
                    warnings.warn(
                        f"[Dataset] Skipping patient {row['Patient_ID']} mod={modality} "
                        f"(attempt {_attempt+1}/10, error: {e})"
                    )
                    new_idx = int(np.random.randint(n_virtual))
                    patient_idx, modality = self._unipair_index[new_idx]
            raise RuntimeError(
                f"[Dataset] Failed to load a valid unipair sample after 10 attempts.")

        for _attempt in range(10):
            try:
                return self._getitem_inner(idx)
            except Exception as e:
                import warnings
                row = self.df.iloc[idx]
                warnings.warn(
                    f"[Dataset] Skipping patient {row['Patient_ID']} "
                    f"(attempt {_attempt+1}/10, error: {e})"
                )
                idx = int(np.random.randint(len(self.df)))
        raise RuntimeError(
            f"[Dataset] Failed to load a valid sample after 10 attempts.")

    def _getitem_unipair(self, patient_idx: int, modality: int):
        """
        Unipair mode: return one of two independent samples from a paired patient.

        modality=0 → CT-only sample:  ct_patch real, pet_patch zeros
        modality=1 → PET-only sample: pet_patch real, ct_patch zeros
        Last element is a scalar modality tensor (0.0 or 1.0).

        Non-MIL return:  (ct_t, pet_t, dose_t, label_t, modality_t)
        MIL return:      (ct_patches, pet_patches, dose_t, label_t, modality_t)
        """
        row   = self.df.iloc[patient_idx]
        pid   = str(row['Patient_ID'])
        label = float(row['label'])
        mod_t = torch.tensor(float(modality), dtype=torch.float32)

        if self.use_mil:
            ct, pet, dose = self._make_patches(pid)   # (N, 1, D, H, W) each

            # Augment only the real modality's patches
            if self.augmentor is not None:
                aug_ct, aug_pet, aug_dose = [], [], []
                N = ct.shape[0]
                for i in range(N):
                    if modality == 0:
                        ct_i, _ = self.augmentor(ct[i], ct[i])
                        pet_i   = ct[i]   # will be zeroed below; shape placeholder
                    else:
                        _, pet_i = self.augmentor(pet[i], pet[i])
                        ct_i    = ct[i]   # will be zeroed below
                    aug_ct.append(ct_i[np.newaxis])
                    aug_pet.append(pet_i[np.newaxis])
                    aug_dose.append(dose[i][np.newaxis])
                ct   = np.concatenate(aug_ct,   axis=0)
                pet  = np.concatenate(aug_pet,  axis=0)
                dose = np.concatenate(aug_dose, axis=0)

            ct_t   = torch.from_numpy(ct.astype(np.float32))
            pet_t  = torch.from_numpy(pet.astype(np.float32))
            dose_t = torch.zeros_like(ct_t)  # zeros for dose in unipair mode

            if modality == 0:   # CT sample: zero out PET
                pet_t = torch.zeros_like(ct_t)
            else:               # PET sample: zero out CT
                ct_t = torch.zeros_like(pet_t)

        else:
            ct, pet, dose = self._make_patch(pid)   # (1, D, H, W) each

            if self.augmentor is not None:
                if modality == 0:
                    ct, _ = self.augmentor(ct, ct)
                else:
                    _, pet = self.augmentor(pet, pet)

            ct_t   = torch.from_numpy(ct.astype(np.float32))
            pet_t  = torch.from_numpy(pet.astype(np.float32))
            dose_t = torch.zeros_like(ct_t)  # zeros for dose in unipair mode

            if modality == 0:   # CT sample: zero out PET
                pet_t = torch.zeros_like(ct_t)
            else:               # PET sample: zero out CT
                ct_t = torch.zeros_like(pet_t)

        lab_t = torch.tensor(label, dtype=torch.float32)
        return (ct_t, pet_t, dose_t, lab_t, mod_t)

    def _getitem_inner(self, idx: int):
        row   = self.df.iloc[idx]
        pid   = str(row['Patient_ID'])
        label = float(row['label'])

        # ── PET-only ablation path ────────────────────────────────────────────
        if self.use_pet_only:
            if self.use_mil:
                _, pet, _ = self._make_patches(pid)   # (N, 1, D, H, W)
                if self.augmentor is not None:
                    aug_pet = []
                    for i in range(len(self.mil_offsets)):
                        _, pet_i = self.augmentor(pet[i], pet[i])
                        aug_pet.append(pet_i[np.newaxis])
                    pet = np.concatenate(aug_pet, axis=0)
            else:
                _, pet, _ = self._make_patch(pid)     # (1, D, H, W)
                if self.augmentor is not None:
                    _, pet = self.augmentor(pet, pet)
                # PET-only MixUp
                if self.augment and np.random.random() < self.mixup_prob:
                    pool = self.pos_idx if label >= 0.5 else self.neg_idx
                    j    = int(np.random.choice(pool))
                    pid2 = str(self.df.iloc[j]['Patient_ID'])
                    lab2 = float(self.df.iloc[j]['label'])
                    _, pet2, _ = self._make_patch(pid2)
                    if self.augmentor is not None:
                        _, pet2 = self.augmentor(pet2, pet2)
                    lam   = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    pet   = lam * pet + (1 - lam) * pet2
                    label = lam * label + (1 - lam) * lab2
            return (
                torch.from_numpy(pet.astype(np.float32)),
                torch.tensor(label, dtype=torch.float32),
            )

        # ── CT-only ablation path ─────────────────────────────────────────────
        if self.use_ct_only:
            if self.use_mil:
                ct, _, _ = self._make_patches(pid)    # (N, 1, D, H, W)
                if self.augmentor is not None:
                    aug_ct = []
                    for i in range(len(self.mil_offsets)):
                        ct_i, _ = self.augmentor(ct[i], ct[i])
                        aug_ct.append(ct_i[np.newaxis])
                    ct = np.concatenate(aug_ct, axis=0)
            else:
                ct, _, _ = self._make_patch(pid)      # (1, D, H, W)
                if self.augmentor is not None:
                    ct, _ = self.augmentor(ct, ct)
                # CT-only MixUp
                if self.augment and np.random.random() < self.mixup_prob:
                    pool = self.pos_idx if label >= 0.5 else self.neg_idx
                    j    = int(np.random.choice(pool))
                    pid2 = str(self.df.iloc[j]['Patient_ID'])
                    lab2 = float(self.df.iloc[j]['label'])
                    ct2, _, _ = self._make_patch(pid2)
                    if self.augmentor is not None:
                        ct2, _ = self.augmentor(ct2, ct2)
                    lam   = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                    ct    = lam * ct + (1 - lam) * ct2
                    label = lam * label + (1 - lam) * lab2
            return (
                torch.from_numpy(ct.astype(np.float32)),
                torch.tensor(label, dtype=torch.float32),
            )

        if self.use_dual_scale:
            # ── Dual-scale path: small patches + large patch, no MixUp ───────
            # Large patch (always single-instance)
            ct_large, dose_large = self._make_large_patch(pid)
            if self.augmentor is not None:
                ct_large, dose_large = self.augmentor(ct_large, dose_large)

            # Small patches (MIL or single-instance, no MixUp)
            if self.use_mil:
                ct, pet, dose = self._make_patches(pid)   # (N, 1, D, H, W)
                if self.augmentor is not None:
                    aug_ct, aug_pet, aug_dose = [], [], []
                    for i in range(len(self.mil_offsets)):
                        ct_i, pet_i = self.augmentor(ct[i], pet[i])
                        aug_ct.append(ct_i[np.newaxis])
                        aug_pet.append(pet_i[np.newaxis])
                        if self.use_dose:
                            dose_i, _ = self.augmentor(dose[i], dose[i])
                            aug_dose.append(dose_i[np.newaxis])
                        else:
                            aug_dose.append(dose[i][np.newaxis])
                    ct   = np.concatenate(aug_ct,   axis=0)
                    pet  = np.concatenate(aug_pet,  axis=0)
                    dose = np.concatenate(aug_dose, axis=0)
            else:
                ct, pet, dose = self._make_patch(pid)     # (1, D, H, W)
                if self.augmentor is not None:
                    ct, pet = self.augmentor(ct, pet)
                    if self.use_dose:
                        dose, _ = self.augmentor(dose, dose)

            return (
                torch.from_numpy(ct.astype(np.float32)),          # (1,D,H,W) or (N,1,D,H,W)
                torch.from_numpy(pet.astype(np.float32)),
                torch.from_numpy(ct_large.astype(np.float32)),    # (1,D_l,H_l,W_l)
                torch.from_numpy(dose_large.astype(np.float32)),
                torch.tensor(label, dtype=torch.float32),
            )

        if self.use_mil:
            # ── MIL path: N instances, no MixUp ──────────────────────────────
            if self.use_bg_patches:
                rng = np.random.default_rng()
                ct, pet, dose, patch_types = self._make_patches_slice_bg(pid, rng)
            else:
                ct, pet, dose = self._make_patches(pid)   # (N, 1, D, H, W)
                patch_types   = None

            if self.augmentor is not None:
                aug_ct, aug_pet, aug_dose = [], [], []
                N = ct.shape[0]
                for i in range(N):
                    # ct[i] has shape (1, D, H, W) — correct 4-D input for augmentor
                    ct_i, pet_i = self.augmentor(ct[i], pet[i])
                    aug_ct.append(ct_i[np.newaxis])    # restore instance dim → (1,1,D,H,W)
                    aug_pet.append(pet_i[np.newaxis])
                    if self.use_dose:
                        dose_i, _ = self.augmentor(dose[i], dose[i])
                        aug_dose.append(dose_i[np.newaxis])
                    else:
                        aug_dose.append(dose[i][np.newaxis])
                ct   = np.concatenate(aug_ct,   axis=0)   # (N, 1, D, H, W)
                pet  = np.concatenate(aug_pet,  axis=0)
                dose = np.concatenate(aug_dose, axis=0)
        else:
            # ── Single-instance path (original) ──────────────────────────────
            ct, pet, dose = self._make_patch(pid)     # (1, D, H, W)
            if self.augmentor is not None:
                ct, pet = self.augmentor(ct, pet)
                if self.use_dose:
                    dose, _ = self.augmentor(dose, dose)

            # MixUp (only in single-instance mode)
            if self.augment and np.random.random() < self.mixup_prob:
                pool = self.pos_idx if label >= 0.5 else self.neg_idx
                j    = int(np.random.choice(pool))
                pid2 = str(self.df.iloc[j]['Patient_ID'])
                lab2 = float(self.df.iloc[j]['label'])
                ct2, pet2, dose2 = self._make_patch(pid2)
                if self.augmentor is not None:
                    ct2, pet2 = self.augmentor(ct2, pet2)
                    if self.use_dose:
                        dose2, _ = self.augmentor(dose2, dose2)
                ct, pet, label = mixup_3d(ct, pet, label, ct2, pet2, lab2,
                                          alpha=self.mixup_alpha)
                if self.use_dose:
                    lam  = label / (lab2 + 1e-8 + label)
                    dose = lam * dose + (1 - lam) * dose2
            patch_types = None

        ct_t   = torch.from_numpy(ct.astype(np.float32))    # (1,D,H,W) or (N,1,D,H,W)
        pet_t  = torch.from_numpy(pet.astype(np.float32))
        dose_t = torch.from_numpy(dose.astype(np.float32))
        lab_t  = torch.tensor(label, dtype=torch.float32)

        # ── Missing gate: compute pet_present and zero pet_t before building tuple ──
        # pet_present must be known before base tuple is assembled so pet_t can be
        # zeroed in-place; pp_tensor is always last in the final return tuple.
        pp_tensor = ()
        if self.use_missing_gate:
            if pid in self._pet_present_cache:
                pet_present = self._pet_present_cache[pid]
            else:
                pet_present = int(os.path.exists(
                    os.path.join(self.pet_dir, f"{pid}_PET.npy")))
            if self.augment and pet_present == 1 and self.pet_dropout_prob > 0:
                if np.random.random() < self.pet_dropout_prob:
                    pet_present = 0
            if pet_present == 0:
                pet_t = torch.zeros_like(pet_t)
            pp_tensor = (torch.tensor(pet_present, dtype=torch.float32),)

        # ── Base tuple: (ct, pet, dose, label [, patch_types]) ────────────────
        if self.use_mil and self.use_bg_patches and patch_types is not None:
            base = (ct_t, pet_t, dose_t, lab_t, torch.from_numpy(patch_types))
        else:
            base = (ct_t, pet_t, dose_t, lab_t)

        # ── Global volumes (inserted between base and pp_tensor) ───────────────
        # Tuple order: base + (ct_global, dose_global) + (pp_t,)
        # Training loop strips pp_t first (if present), then ct_global/dose_global.
        global_tensors = ()
        if self.use_global_branch:
            ct_g, dose_g = self._load_global(pid)
            global_tensors = (ct_g, dose_g)

        return base + global_tensors + pp_tensor

    @property
    def label_list(self) -> List[int]:
        base = [int(round(l)) for l in self.labels]
        if self.use_unipair:
            # Each virtual sample maps to a patient via _unipair_index
            return [base[i] for i, _ in self._unipair_index]
        return base
