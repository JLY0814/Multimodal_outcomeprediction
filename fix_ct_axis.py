"""
fix_ct_axis.py
Detect and fix CT .npy files stored in (H, W, Z) order instead of (Z, H, W).

Detection rule:
  shape[0] > shape[2]  →  Z-axis is at the end  →  needs fix
  shape[0] < shape[2]  →  Z-axis is at the front →  already correct

Fix:
  np.transpose(arr, (2, 1, 0))  :  (X, Y, Z) → (Z, Y, X)

Usage:
  DRY_RUN = True   → only print what would be changed, no files written
  DRY_RUN = False  → fix in-place (original saved to {file}_backup.npy)
"""

import os
import numpy as np
import pandas as pd

# ── Hard-coded paths ──────────────────────────────────────────────────────────
CSV    = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"

# ── Options ───────────────────────────────────────────────────────────────────
DRY_RUN = True    # set False to actually write corrected files
BACKUP  = True    # keep original as {pid}_CT_backup.npy before overwriting

# Scan all patients in CSV, or restrict to CT-only (no PET file)
CT_ONLY_ONLY = False   # False = scan ALL patients;  True = only CT-only patients

# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV)
df['Patient_ID'] = df['Patient_ID'].astype(str)
pids = df['Patient_ID'].tolist()

if CT_ONLY_ONLY:
    PET_DIR = CT_DIR   # PET files are in the same directory
    pids = [p for p in pids
            if not os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy"))]
    print(f"Scanning CT-only patients: {len(pids)}")
else:
    print(f"Scanning all patients: {len(pids)}")

n_ok    = 0
n_wrong = 0
n_skip  = 0

for pid in pids:
    path = os.path.join(CT_DIR, f"{pid}_CT.npy")
    if not os.path.exists(path):
        n_skip += 1
        continue

    arr = np.load(path, mmap_mode='r')
    shape = arr.shape

    if len(shape) != 3:
        print(f"  [SKIP] {pid}  unexpected ndim={len(shape)}  shape={shape}")
        n_skip += 1
        continue

    if shape[0] <= shape[2]:
        # Z is already at front (or ambiguous square case)
        n_ok += 1
        continue

    # shape[0] > shape[2]  →  Z is at the end, needs fix
    n_wrong += 1
    new_shape = (shape[2], shape[0], shape[1])
    print(f"  [FIX]  {pid:<24s}  {shape}  →  {new_shape}")

    if not DRY_RUN:
        arr_full = np.load(path)          # load fully (not mmap) before writing
        arr_fixed = np.transpose(arr_full, (2, 1, 0)).astype(arr_full.dtype)

        if BACKUP:
            backup_path = os.path.join(CT_DIR, f"{pid}_CT_backup.npy")
            np.save(backup_path, arr_full)

        np.save(path, arr_fixed)

print()
print("=" * 50)
print(f"  Total scanned : {n_ok + n_wrong + n_skip}")
print(f"  Already correct : {n_ok}")
print(f"  Need fix        : {n_wrong}")
print(f"  Missing / skip  : {n_skip}")
if DRY_RUN:
    print()
    print("  DRY_RUN=True — no files were modified.")
    print("  Set DRY_RUN=False to apply fixes.")
else:
    print(f"  Fixed           : {n_wrong}  (backups={'saved' if BACKUP else 'skipped'})")
