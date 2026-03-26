import numpy as np
import os
import re
import csv
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
PET_FOLDER  = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all\registered_all_resampled_09766_3mm"
MASK_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all\test_all\test_all_1\masks_selected"
EXCEL_PATH  = r"Z:\Active\Patient list and data for Data Collection-2025\Cervical\2022 Cervix Database_SM_2025Jan22 pull for HLi-from Markovina-2025 01 22.xlsx"
OUTPUT_FOLDER = os.path.join(PET_FOLDER, "normalized_PET")   # new folder, auto-created

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — PET + Mask  →  mean PET value per patient (all masks)
# ═══════════════════════════════════════════════════════════════════════════════

pet_files  = [f for f in os.listdir(PET_FOLDER)  if f.endswith(".npy")]
mask_files = [f for f in os.listdir(MASK_FOLDER) if f.endswith(".npy") and "cervix" in f.lower()]

print(f"Found {len(pet_files)} PET files")
print(f"Found {len(mask_files)} mask files (containing 'cervix')\n")

def extract_id(filename):
    match = re.search(r"(GYNDATASET\d+)", filename, re.IGNORECASE)
    return match.group(1).upper() if match else None

pet_map  = {}          # { "GYNDATASET003": full_path }
mask_map = {}          # { "GYNDATASET003": [path1, path2, ...] }

for f in pet_files:
    gid = extract_id(f)
    if gid:
        pet_map[gid] = os.path.join(PET_FOLDER, f)

for f in mask_files:
    gid = extract_id(f)
    if gid:
        mask_map.setdefault(gid, []).append(os.path.join(MASK_FOLDER, f))

common_ids = sorted(set(pet_map.keys()) & set(mask_map.keys()))
print(f"Matched PET-Mask pairs: {len(common_ids)}\n")

# mean_results[gid] = { mask_filename: mean_pet_value, ... }
mean_results = {}

print("── PET Mask Overlay Results ─────────────────────────────────────────────")
print(f"{'ID':<18} {'Mask File':<55} {'Voxels':<10} {'Mean PET'}")
print("-" * 100)

for gid in common_ids:
    pet = np.load(pet_map[gid])

    for mask_path in mask_map[gid]:
        mask       = np.load(mask_path)
        mask_fname = os.path.basename(mask_path)

        if pet.shape != mask.shape:
            print(f"{gid:<18} {mask_fname:<55} {'SHAPE MISMATCH – skipped'}")
            continue

        binary_mask = mask.astype(bool)
        num_voxels  = binary_mask.sum()

        if num_voxels == 0:
            print(f"{gid:<18} {mask_fname:<55} {'0':<10} {'Empty mask – skipped'}")
            continue

        mean_val = float(pet[binary_mask].mean())
        print(f"{gid:<18} {mask_fname:<55} {num_voxels:<10} {mean_val:.4f}")
        mean_results.setdefault(gid, {})[mask_fname] = mean_val

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Pick shortest-name mask per patient, read Excel, compute ratio
# ═══════════════════════════════════════════════════════════════════════════════
#
#   Excel layout:
#       Row 1  → headers        (one column header is "Cervix SUV")
#       Row 2  → GYNDATASET001
#       Row 3  → GYNDATASET002  ...
#   Excel row N  →  GYNDATASET{N-1:03d}
#   Missing PET patients are skipped; order never shifts.
#
# ═══════════════════════════════════════════════════════════════════════════════

# For each patient, select the mask with the shortest filename
# selected_mask[gid] = (mask_fname, mean_pet_value)
selected_mask = {}
for gid, masks in mean_results.items():
    best_fname = min(masks.keys(), key=len)          # shortest name; ties → min picks one
    selected_mask[gid] = (best_fname, masks[best_fname])

df = pd.read_excel(EXCEL_PATH, header=0)

# Find "Cervix SUV" column
cervix_suv_col = None
for col in df.columns:
    if str(col).strip() == "Cervix SUV":
        cervix_suv_col = col
        break

if cervix_suv_col is None:
    print("\n[ERROR] Could not find a column with header 'Cervix SUV'.")
    print("Available columns:", list(df.columns))
    exit()

print(f"\n── Excel Info ────────────────────────────────────────────────────────────")
print(f"Cervix SUV column found: '{cervix_suv_col}'")
print(f"Total data rows in Excel: {len(df)}\n")

# ratio_map[gid] = ratio value (only for valid entries)
ratio_map    = {}
ratio_results = []

print("── Ratio Results (shortest-name mask per patient) ───────────────────────")
print(f"{'GYNDATASET ID':<18} {'Cervix SUV':<14} {'Mean PET':<14} {'Ratio':<18} {'Selected Mask'}")
print("-" * 100)

for row_idx in range(len(df)):
    gyn_num = row_idx + 1
    gid     = f"GYNDATASET{gyn_num:03d}"
    suv_val = df.iloc[row_idx][cervix_suv_col]

    if gid not in selected_mask:
        print(f"{gid:<18} — no PET data, skipped")
        continue

    mask_fname, mean_pet = selected_mask[gid]

    # Validate SUV
    try:
        suv_float = float(suv_val)
    except (ValueError, TypeError):
        print(f"{gid:<18} {'SUV invalid':<14} {mean_pet:<14.4f} {'— skipped':<18} {mask_fname}")
        ratio_results.append({
            "GYNDATASET_ID": gid, "Cervix_SUV": suv_val,
            "Mean_PET": mean_pet, "Ratio": None,
            "Selected_Mask": mask_fname, "Note": "Cervix SUV invalid or missing"
        })
        continue

    if suv_float == 0:
        print(f"{gid:<18} {'0':<14} {mean_pet:<14.4f} {'— div by 0':<18} {mask_fname}")
        ratio_results.append({
            "GYNDATASET_ID": gid, "Cervix_SUV": 0,
            "Mean_PET": mean_pet, "Ratio": None,
            "Selected_Mask": mask_fname, "Note": "Cervix SUV is 0 (division by zero)"
        })
        continue

    ratio = mean_pet / suv_float
    ratio_map[gid] = ratio
    print(f"{gid:<18} {suv_float:<14.4f} {mean_pet:<14.4f} {ratio:<18.4f} {mask_fname}")

    ratio_results.append({
        "GYNDATASET_ID": gid, "Cervix_SUV": suv_float,
        "Mean_PET": mean_pet, "Ratio": ratio,
        "Selected_Mask": mask_fname, "Note": ""
    })

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Normalize PET arrays  (PET / ratio)  →  save to new folder
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print(f"\n── Normalizing PET files → {OUTPUT_FOLDER}")
print(f"{'GYNDATASET ID':<18} {'Ratio Used':<18} {'Original File':<55} {'Saved As'}")
print("-" * 120)

normalized_count = 0

for gid, ratio in ratio_map.items():
    if gid not in pet_map:
        continue                                        # shouldn't happen, but safe

    pet          = np.load(pet_map[gid])
    normalized   = pet / ratio                          # element-wise division

    # Keep the original filename as-is for the output
    out_fname    = os.path.basename(pet_map[gid])
    out_path     = os.path.join(OUTPUT_FOLDER, out_fname)
    np.save(out_path, normalized)

    print(f"{gid:<18} {ratio:<18.4f} {out_fname:<55} {out_fname}")
    normalized_count += 1

print(f"\n✓ Normalized {normalized_count} PET files saved to: {OUTPUT_FOLDER}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Save ratio results to CSV
# ═══════════════════════════════════════════════════════════════════════════════

csv_path = os.path.join(PET_FOLDER, "pet_mask_ratio_results.csv")
fieldnames = ["GYNDATASET_ID", "Cervix_SUV", "Mean_PET", "Ratio", "Selected_Mask", "Note"]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(ratio_results)

print(f"✓ Ratio results CSV saved to: {csv_path}")
