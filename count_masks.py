"""
count_masks.py
Count patients who have at least one MTV Cervix mask in MASK_DIR.
Definition: a mask file whose name contains both 'mtv' and 'cervix'
(case-insensitive). One patient may have multiple such files; counted once.
"""

import os
import glob
import pandas as pd
from dataset import parse_labels

CSV      = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/all_patients_info.csv"
CT_DIR   = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
PET_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
MASK_DIR = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy/npy_masks_MTV_PTV_resampled"
LABEL_COL = "recurrence"

# ── Build patient → [matched files] index ────────────────────────────────────

all_npy = glob.glob(os.path.join(MASK_DIR, "*.npy"))
print(f"Total .npy files in directory: {len(all_npy)}")

# Collect all files that contain both 'mtv' and 'cervix' in filename
matched = [p for p in all_npy
           if "mtv" in os.path.basename(p).lower()
           and "cervix" in os.path.basename(p).lower()]

print(f"Files matching mtv+cervix rule : {len(matched)}")
print(f"\nExample matched filenames (first 10):")
for p in sorted(matched)[:10]:
    print(f"  {os.path.basename(p)}")

# ── Map to patient IDs ────────────────────────────────────────────────────────
# patient_id = longest prefix of filename (before '.npy') that matches a known pid
df = pd.read_csv(CSV)
df = parse_labels(df, label_col=LABEL_COL)
df["Patient_ID"] = df["Patient_ID"].astype(str)
df["has_ct"]  = df["Patient_ID"].apply(lambda p: os.path.exists(os.path.join(CT_DIR,  f"{p}_CT.npy")))
df["has_pet"] = df["Patient_ID"].apply(lambda p: os.path.exists(os.path.join(PET_DIR, f"{p}_PET.npy")))
df = df[df["has_ct"]].copy().reset_index(drop=True)

# For each patient, check if any matched file starts with their pid (case-insensitive)
pid_lower_set = {pid.lower(): pid for pid in df["Patient_ID"]}

patients_with_mask = set()
for path in matched:
    fname_lower = os.path.basename(path).lower()
    for pid_lower in pid_lower_set:
        if fname_lower.startswith(pid_lower):
            patients_with_mask.add(pid_lower)
            break

df["has_mask"] = df["Patient_ID"].apply(lambda p: p.lower() in patients_with_mask)

# ── Report ────────────────────────────────────────────────────────────────────

def report(group, label):
    n        = len(group)
    n_mask   = group["has_mask"].sum()
    n_nomask = n - n_mask
    print(f"\n{label}")
    print(f"  Total       : {n}")
    print(f"  Has mask    : {n_mask}  ({100*n_mask/max(n,1):.1f}%)")
    print(f"  No mask     : {n_nomask}  ({100*n_nomask/max(n,1):.1f}%)")

paired = df[df["has_pet"]]
ctonly = df[~df["has_pet"]]

print()
report(paired, "Paired (CT+PET)")
report(ctonly,  "CT-only")
report(df,      "All")
