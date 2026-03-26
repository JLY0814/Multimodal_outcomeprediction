import os
import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib


# ─── Paths ────────────────────────────────────────────────────────────────────
PET_FOLDER  = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all\registered_all_resampled_09766_3mm"
MASK_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\masks_MTV_PTV_resampled_0.9766_3.0mm"
EXCEL_PATH  = r"Z:\Active\Patient list and data for Data Collection-2025\Cervical\2022 Cervix Database_SM_2025Jan22 pull for HLi-from Markovina-2025 01 22.xlsx"

OUTPUT_FOLDER = os.path.join(PET_FOLDER, "normalized_PET_nii")  # new folder, auto-created


# ─── Helpers ─────────────────────────────────────────────────────────────────
def extract_id(filename: str):
    m = re.search(r"(GYNDATASET\d+)", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def is_mask_file(name: str) -> bool:
    name_l = name.lower()
    return (
        (name_l.endswith(".nii") or name_l.endswith(".nii.gz"))
        and ("cervix" in name_l)
    )


def load_nii_data(path: str, dtype=np.float32):
    """
    Load NIfTI voxel data as numpy array and return (array, affine, header).
    Using get_fdata gives scaled values if slope/intercept exist.
    """
    nii = nib.load(path)
    arr = nii.get_fdata(dtype=dtype)
    return arr, nii.affine, nii.header


def save_nii_data(path: str, arr: np.ndarray, affine):
    """
    Save array to NIfTI with scl_slope/inter disabled to avoid extra scaling later.
    """
    out = nib.Nifti1Image(arr, affine)
    out.header["scl_slope"] = 1
    out.header["scl_inter"] = 0
    nib.save(out, path)


def get_done_ids(output_folder: str) -> set:
    """
    If OUTPUT_FOLDER already contains GYNDATASETxxx_PET.nii(.gz),
    we consider that patient done and skip all processing for it.
    """
    done = set()
    if not os.path.isdir(output_folder):
        return done

    for f in os.listdir(output_folder):
        fl = f.lower()
        if not fl.endswith((".nii", ".nii.gz")):
            continue
        if not (fl.endswith("_pet.nii") or fl.endswith("_pet.nii.gz")):
            continue
        gid = extract_id(f)
        if gid:
            done.add(gid)
    return done


# ═══════════════════════════════════════════════════════════════════════════════
# PREP — Build maps, and decide TODO list (skip already processed)
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
done_ids = get_done_ids(OUTPUT_FOLDER)

pet_files  = [
    f for f in os.listdir(PET_FOLDER)
    if f.lower().endswith((".nii", ".nii.gz")) and f.lower().endswith(("_pet.nii", "_pet.nii.gz"))
]
mask_files = [f for f in os.listdir(MASK_FOLDER) if is_mask_file(f)]

print(f"Found {len(pet_files)} PET files (*_PET.nii/.nii.gz)")
print(f"Found {len(mask_files)} mask files (containing 'cervix')")
print(f"Found {len(done_ids)} already-normalized PET files in OUTPUT_FOLDER (will skip them)\n")

pet_map  = {}   # { "GYNDATASET003": full_path }
mask_map = {}   # { "GYNDATASET003": [path1, path2, ...] }

for f in pet_files:
    gid = extract_id(f)
    if gid:
        pet_map[gid] = os.path.join(PET_FOLDER, f)

for f in mask_files:
    gid = extract_id(f)
    if gid:
        mask_map.setdefault(gid, []).append(os.path.join(MASK_FOLDER, f))

common_ids_all = sorted(set(pet_map.keys()) & set(mask_map.keys()))
todo_ids = [gid for gid in common_ids_all if gid not in done_ids]

print(f"Matched PET-Mask pairs (all): {len(common_ids_all)}")
print(f"To process (not done yet):     {len(todo_ids)}\n")

if len(todo_ids) == 0:
    print("✓ Nothing to do. All matched patients already have outputs in OUTPUT_FOLDER.")
    raise SystemExit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — PET + Mask  →  max PET value inside mask per patient (all masks)
# (ONLY for TODO patients)
# ═══════════════════════════════════════════════════════════════════════════════

max_results = {}  # max_results[gid] = { mask_filename: max_pet_value, ... }

print("── PET Mask Overlay Results (TODO only) ─────────────────────────────────")
print(f"{'ID':<18} {'Mask File':<60} {'Voxels':<12} {'Max PET'}")
print("-" * 110)

for gid in todo_ids:
    pet_path = pet_map[gid]
    pet, pet_affine, _ = load_nii_data(pet_path, dtype=np.float32)

    for mask_path in mask_map[gid]:
        mask, _, _ = load_nii_data(mask_path, dtype=np.float32)
        mask_fname = os.path.basename(mask_path)

        if pet.shape != mask.shape:
            print(f"{gid:<18} {mask_fname:<60} {'SHAPE MISMATCH – skipped'}")
            continue

        binary_mask = mask > 0
        num_voxels = int(binary_mask.sum())

        if num_voxels == 0:
            print(f"{gid:<18} {mask_fname:<60} {0:<12} {'Empty mask – skipped'}")
            continue

        max_val = float(np.max(pet[binary_mask]))
        print(f"{gid:<18} {mask_fname:<60} {num_voxels:<12} {max_val:.4f}")
        max_results.setdefault(gid, {})[mask_fname] = max_val


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Pick shortest-name mask per patient, read Excel, compute ratio
# (ONLY for TODO patients that have max_results)
# ═══════════════════════════════════════════════════════════════════════════════

selected_mask = {}
for gid, masks in max_results.items():
    if not masks:
        continue
    best_fname = min(masks.keys(), key=len)
    selected_mask[gid] = (best_fname, masks[best_fname])

df = pd.read_excel(EXCEL_PATH, header=0)

cervix_suv_col = None
for col in df.columns:
    if str(col).strip() == "Cervix SUV":
        cervix_suv_col = col
        break

if cervix_suv_col is None:
    print("\n[ERROR] Could not find a column with header 'Cervix SUV'.")
    print("Available columns:", list(df.columns))
    raise SystemExit(1)

print(f"\n── Excel Info ────────────────────────────────────────────────────────────")
print(f"Cervix SUV column found: '{cervix_suv_col}'")
print(f"Total data rows in Excel: {len(df)}\n")

ratio_map = {}
ratio_results = []

print("── Ratio Results (shortest-name mask per patient, TODO only) ─────────────")
print(f"{'GYNDATASET ID':<18} {'Cervix SUV':<14} {'Max PET':<14} {'Ratio':<18} {'Selected Mask'}")
print("-" * 110)

for row_idx in range(len(df)):
    gyn_num = row_idx + 1
    gid = f"GYNDATASET{gyn_num:03d}"

    # skip patients already processed
    if gid in done_ids:
        continue

    suv_val = df.iloc[row_idx][cervix_suv_col]

    if gid not in selected_mask:
        # either no PET/mask match or max_results empty
        continue

    mask_fname, max_pet = selected_mask[gid]

    try:
        suv_float = float(suv_val)
    except (ValueError, TypeError):
        ratio_results.append({
            "GYNDATASET_ID": gid, "Cervix_SUV": suv_val,
            "Max_PET": max_pet, "Ratio": None,
            "Selected_Mask": mask_fname, "Note": "Cervix SUV invalid or missing"
        })
        continue

    if suv_float == 0:
        ratio_results.append({
            "GYNDATASET_ID": gid, "Cervix_SUV": 0,
            "Max_PET": max_pet, "Ratio": None,
            "Selected_Mask": mask_fname, "Note": "Cervix SUV is 0 (division by zero)"
        })
        continue

    if max_pet == 0:
        ratio_results.append({
            "GYNDATASET_ID": gid, "Cervix_SUV": suv_float,
            "Max_PET": max_pet, "Ratio": None,
            "Selected_Mask": mask_fname, "Note": "Max PET inside mask is 0 (cannot scale)"
        })
        continue

    ratio = max_pet / suv_float
    ratio_map[gid] = ratio

    print(f"{gid:<18} {suv_float:<14.4f} {max_pet:<14.4f} {ratio:<18.4f} {mask_fname}")

    ratio_results.append({
        "GYNDATASET_ID": gid, "Cervix_SUV": suv_float,
        "Max_PET": max_pet, "Ratio": ratio,
        "Selected_Mask": mask_fname, "Note": ""
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Normalize PET volumes (PET / ratio) → save as NIfTI in new folder
# (ONLY for TODO patients with valid ratio)
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n── Normalizing PET files (TODO only) → {OUTPUT_FOLDER}")
print(f"{'GYNDATASET ID':<18} {'Ratio Used':<18} {'Original File':<55} {'Saved As'}")
print("-" * 120)

normalized_count = 0

for gid, ratio in ratio_map.items():
    # extra safety: if output already exists, skip
    out_candidate = find_existing = None

    if gid not in pet_map:
        continue

    pet_path = pet_map[gid]
    out_fname = Path(pet_path).name  # keep the original filename
    out_path = os.path.join(OUTPUT_FOLDER, out_fname)

    if os.path.exists(out_path):
        # should not happen if done_ids logic is correct, but keep it safe
        continue

    pet, pet_affine, _ = load_nii_data(pet_path, dtype=np.float32)
    normalized = pet / float(ratio)

    save_nii_data(out_path, normalized.astype(np.float32), pet_affine)

    print(f"{gid:<18} {ratio:<18.4f} {out_fname:<55} {out_fname}")
    normalized_count += 1

print(f"\n✓ Normalized {normalized_count} new PET NIfTI files saved to: {OUTPUT_FOLDER}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Save ratio results to CSV (THIS RUN results only)
# ═══════════════════════════════════════════════════════════════════════════════

csv_path = os.path.join(PET_FOLDER, "pet_mask_ratio_results_nii_max.csv")
fieldnames = ["GYNDATASET_ID", "Cervix_SUV", "Max_PET", "Ratio", "Selected_Mask", "Note"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ratio_results)

print(f"✓ Ratio results CSV saved to: {csv_path}")

