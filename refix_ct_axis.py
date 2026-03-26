"""
refix_ct_axis.py
Re-transpose CT files from backup using the correct (2,1,0) order,
overwriting the previously incorrectly transposed files.

Finds all *_CT_backup.npy files, applies np.transpose(arr, (2,1,0)),
and saves to the corresponding *_CT.npy (overwrite).
"""

import os
import glob
import numpy as np

CT_DIR  = "/shared/anastasio-s3/jyue/Large-Scale-Medical/longi_test_data_npy"
DRY_RUN = True   # set False to actually write

backups = sorted(glob.glob(os.path.join(CT_DIR, "*_CT_backup.npy")))
print(f"Found {len(backups)} backup files.\n")

for backup_path in backups:
    target_path = backup_path.replace("_CT_backup.npy", "_CT.npy")
    arr = np.load(backup_path, mmap_mode='r')
    arr_fixed_shape = (arr.shape[2], arr.shape[1], arr.shape[0])
    print(f"  {os.path.basename(backup_path)}")
    print(f"    backup shape : {arr.shape}  →  fixed shape : {arr_fixed_shape}")

    if not DRY_RUN:
        arr_full  = np.load(backup_path)
        arr_fixed = np.transpose(arr_full, (2, 1, 0)).astype(arr_full.dtype)
        np.save(target_path, arr_fixed)
        print(f"    saved → {target_path}")

print()
if DRY_RUN:
    print("DRY_RUN=True — no files written. Set DRY_RUN=False to apply.")
else:
    print(f"Done. {len(backups)} files overwritten.")
