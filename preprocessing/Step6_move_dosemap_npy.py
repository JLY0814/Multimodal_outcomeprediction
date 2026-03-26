from pathlib import Path
import shutil

# ======================
# Paths (edit if needed)
# ======================
MASK_SRC_DIR = Path(r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\dose_npy_resampled_to_ct")

DATA_DIR = Path(r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all\test_all\test_all_1")

# New folder to store the moved/copied masks
DEST_MASK_DIR = DATA_DIR / "dosemap_selected"

# ======================
# Behavior settings
# ======================
MOVE_FILES = True   # True = move (cut), False = copy
REQUIRE_BOTH_CT_PET = False  # True = only process patients that have BOTH _CT.npy and _PET.npy

# If destination file already exists:
# "skip" = keep existing, do nothing
# "rename" = move/copy but auto-rename the incoming file
ON_CONFLICT = "rename"  # "skip" or "rename"


def patient_id_from_data_file(p: Path) -> str | None:
    """Extract patient id like 'GYNDATASET003' from 'GYNDATASET003_CT.npy' / 'GYNDATASET003_PET.npy'."""
    name = p.name
    if name.endswith("_CT.npy"):
        return name[:-len("_CT.npy")]
    if name.endswith("_PET.npy"):
        return name[:-len("_PET.npy")]
    return None


def safe_dest_path(dest_dir: Path, filename: str) -> Path:
    """Return a destination path that avoids overwriting if ON_CONFLICT=='rename'."""
    out = dest_dir / filename
    if not out.exists():
        return out

    if ON_CONFLICT == "skip":
        return out  # caller will interpret as skip

    stem = out.stem
    suffix = out.suffix
    k = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{k}{suffix}"
        if not candidate.exists():
            return candidate
        k += 1


def main():
    if not MASK_SRC_DIR.exists():
        raise FileNotFoundError(f"Mask source folder not found: {MASK_SRC_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    DEST_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Collect patient IDs present in DATA_DIR based on *_CT.npy / *_PET.npy
    ct_patients = set()
    pet_patients = set()
    all_patients = set()

    for f in DATA_DIR.glob("*.npy"):
        pid = patient_id_from_data_file(f)
        if pid is None:
            continue
        all_patients.add(pid)
        if f.name.endswith("_CT.npy"):
            ct_patients.add(pid)
        elif f.name.endswith("_PET.npy"):
            pet_patients.add(pid)

    if REQUIRE_BOTH_CT_PET:
        patients = sorted(ct_patients.intersection(pet_patients))
    else:
        patients = sorted(all_patients)

    print(f"Found {len(patients)} patient(s) in DATA_DIR.")
    if REQUIRE_BOTH_CT_PET:
        print("Mode: REQUIRE_BOTH_CT_PET=True (only patients with both CT & PET)")

    # 2) For each patient, move/copy matching masks
    total_masks_found = 0
    total_moved_or_copied = 0
    total_skipped_conflict = 0

    for pid in patients:
        # masks like: GYNDATASET003_*.npy
        matches = list(MASK_SRC_DIR.glob(f"{pid}_*.npy"))

        if not matches:
            # If your mask files sometimes have no ".npy" extension, uncomment this:
            # matches = [p for p in MASK_SRC_DIR.glob(f"{pid}_*") if p.is_file()]
            print(f"[{pid}] No mask files found in {MASK_SRC_DIR}")
            continue

        total_masks_found += len(matches)
        print(f"[{pid}] {len(matches)} mask(s) found.")

        for src in matches:
            dest = safe_dest_path(DEST_MASK_DIR, src.name)

            # conflict handling
            if dest.exists() and ON_CONFLICT == "skip":
                total_skipped_conflict += 1
                print(f"  - SKIP (exists): {src.name}")
                continue

            if MOVE_FILES:
                shutil.move(str(src), str(dest))
                print(f"  - MOVED  -> {dest.name}")
            else:
                shutil.copy2(str(src), str(dest))
                print(f"  - COPIED -> {dest.name}")

            total_moved_or_copied += 1

    print("\nDone.")
    print(f"Destination folder: {DEST_MASK_DIR}")
    print(f"Total masks found: {total_masks_found}")
    print(f"Total {'moved' if MOVE_FILES else 'copied'}: {total_moved_or_copied}")
    if ON_CONFLICT == "skip":
        print(f"Total skipped due to name conflicts: {total_skipped_conflict}")


if __name__ == "__main__":
    main()
