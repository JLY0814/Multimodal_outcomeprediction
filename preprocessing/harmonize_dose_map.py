
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# w
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List
import csv

import pydicom
from pydicom.uid import generate_uid


# =========================
# CONFIG
# =========================
ROOT_DIR = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data"
OUT_PREFIX = "RD_RxNorm_"
SKIP_LOG_NAME = "skip_reasons.txt"

# NEW: fallback maxGy CSV (hard-coded as requested)
MAXGY_CSV_PATH = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\dose_harmonize_details.csv"

# NEW: "reasonable range" for using maxGy as Rx proxy (tune if you want)
MAXGY_REASONABLE_MIN = 30.0
MAXGY_REASONABLE_MAX = 90.0


def dicom_read(path: Path):
    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


def safe_get(ds, attr: str, default=None):
    return getattr(ds, attr, default)


def is_rtdose(ds) -> bool:
    return str(safe_get(ds, "SOPClassUID", "")) == "1.2.840.10008.5.1.4.1.1.481.2"


def is_rtplan(ds) -> bool:
    return str(safe_get(ds, "SOPClassUID", "")) == "1.2.840.10008.5.1.4.1.1.481.5"


def extract_referenced_rp_uid_from_rd(rd_ds) -> Optional[str]:
    seq = safe_get(rd_ds, "ReferencedRTPlanSequence", None)
    if not seq:
        return None
    try:
        item0 = seq[0]
        uid = safe_get(item0, "ReferencedSOPInstanceUID", None)
        return str(uid) if uid else None
    except Exception:
        return None


def extract_rx_from_rp(rp_ds):
    """
    Robust Rx extraction from RTPLAN.

    Returns (rx_dose_gy, n_fractions, notes)
    """
    notes = []
    rx_dose_gy = None
    n_fractions = None

    # Fractions planned
    fgs = getattr(rp_ds, "FractionGroupSequence", None)
    if fgs and len(fgs) > 0:
        nf = getattr(fgs[0], "NumberOfFractionsPlanned", None)
        try:
            if nf is not None:
                n_fractions = int(nf)
                notes.append(f"FractionsPlanned={n_fractions}")
        except Exception:
            notes.append(f"FractionsPlanned parse failed: {nf}")
    else:
        notes.append("FractionsPlanned=NOT_FOUND")

    drs = getattr(rp_ds, "DoseReferenceSequence", None)
    if not drs:
        notes.append("DoseReferenceSequence=NOT_FOUND")
        return None, n_fractions, notes

    def collect(items, label):
        cands = []
        for it in items:
            tpd = getattr(it, "TargetPrescriptionDose", None)
            if tpd is not None:
                try:
                    cands.append((float(tpd), f"{label}:TargetPrescriptionDose"))
                    continue
                except Exception:
                    pass
            for name in ["DoseReferenceDose", "DeliveryMaximumDose", "TargetMaximumDose"]:
                val = getattr(it, name, None)
                if val is None:
                    continue
                try:
                    cands.append((float(val), f"{label}:{name}"))
                except Exception:
                    pass
        return cands

    # TARGET items first
    target_items = []
    for it in drs:
        ref_type = str(getattr(it, "DoseReferenceType", "")).upper().strip()
        if ref_type == "TARGET":
            target_items.append(it)

    candidates = []
    if target_items:
        candidates.extend(collect(target_items, "TARGET"))
        notes.append(f"TARGET_items={len(target_items)}")
    else:
        notes.append("TARGET_items=0")

    # fallback to ANY
    if not candidates:
        any_items = list(drs)
        candidates.extend(collect(any_items, "ANY"))
        notes.append("Fallback to ANY dose reference items")

    if not candidates:
        notes.append("RxDoseGy=NOT_FOUND (no usable prescription/max dose fields in DoseReferenceSequence)")
        return None, n_fractions, notes

    rx_dose_gy, src = max(candidates, key=lambda x: x[0])
    notes.append(f"RxDoseGy={rx_dose_gy} from {src}")
    return rx_dose_gy, n_fractions, notes


def load_patient_maxgy_map(csv_path: str) -> Tuple[Dict[str, float], str]:
    """
    Load patient->maxGy mapping from dose_harmonize_details.csv.

    Returns (map, info_message).
    The loader tries to auto-detect patient id column and maxGy column.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}, f"CSV not found: {path}"

    # candidate column names (case-insensitive)
    patient_keys = {
        "patient", "patientid", "patient_id", "patient folder", "patient_folder",
        "patientfolder", "folder", "patientname"
    }
    maxgy_keys = {
        "maxgy", "max_gy", "dose_maxgy", "dosestats_maxgy", "max(g y)", "max(gy)", "max gy"
    }

    mp: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}, "CSV has no header/fieldnames."

        # map fieldnames to normalized form
        norm = {name: name.strip().lower().replace("__", "_") for name in reader.fieldnames}

        patient_col = None
        maxgy_col = None

        for orig, n in norm.items():
            if n in patient_keys:
                patient_col = orig
            if n in maxgy_keys:
                maxgy_col = orig

        if patient_col is None or maxgy_col is None:
            return {}, (
                f"CSV columns not recognized. Found columns={reader.fieldnames}. "
                f"Need patient column in {sorted(patient_keys)} and maxGy column in {sorted(maxgy_keys)}."
            )

        for row in reader:
            pid = (row.get(patient_col, "") or "").strip()
            if not pid:
                continue
            v = (row.get(maxgy_col, "") or "").strip()
            if not v:
                continue
            try:
                mp[pid] = float(v)
            except Exception:
                continue

    return mp, f"Loaded maxGy for {len(mp)} patients from {path.name} (patient_col='{patient_col}', maxGy_col='{maxgy_col}')."


def make_rx_normalized_rtdose(
    rd_path: Path,
    rp_path: Path,
    out_path: Path,
    patient_id: str,
    maxgy_map: Dict[str, float],
) -> Tuple[bool, str]:
    """
    Create a new RTDOSE where DoseGridScaling is divided by RxDose (from RTPLAN),
    but if RxDose not found in RTPLAN, fallback to maxGy from CSV if reasonable.
    """
    try:
        rd_ds = pydicom.dcmread(str(rd_path), stop_before_pixels=True, force=True)
        rp_ds = pydicom.dcmread(str(rp_path), stop_before_pixels=True, force=True)
    except Exception as e:
        return False, f"Read failed: {type(e).__name__}: {e}"

    if not is_rtdose(rd_ds):
        return False, "Input RD is not RTDOSE SOPClassUID."
    if not is_rtplan(rp_ds):
        return False, "Input RP is not RTPLAN SOPClassUID."

    rx_dose_gy, n_frac, notes = extract_rx_from_rp(rp_ds)

    rx_source = "RTPLAN"
    if rx_dose_gy is None or rx_dose_gy <= 0:
        # NEW: fallback to CSV maxGy
        fallback_maxgy = maxgy_map.get(patient_id, None)
        if fallback_maxgy is None:
            return False, f"RxDoseGy not found in RTPLAN AND maxGy missing in CSV for patient={patient_id}. Notes: {'; '.join(notes)}"

        if not (MAXGY_REASONABLE_MIN <= float(fallback_maxgy) <= MAXGY_REASONABLE_MAX):
            return False, (
                f"RxDoseGy not found in RTPLAN AND maxGy={fallback_maxgy} out of reasonable range "
                f"[{MAXGY_REASONABLE_MIN},{MAXGY_REASONABLE_MAX}] for patient={patient_id}. Notes: {'; '.join(notes)}"
            )

        rx_dose_gy = float(fallback_maxgy)
        rx_source = "CSV:maxGy"
        notes.append(f"Fallback RxDoseGy={rx_dose_gy} from CSV maxGy (patient={patient_id})")

    # Need DoseGridScaling
    dgs = safe_get(rd_ds, "DoseGridScaling", None)
    if dgs is None:
        return False, "DoseGridScaling missing in RD."

    try:
        dgs_f = float(dgs)
    except Exception:
        return False, f"DoseGridScaling not numeric: {dgs}"

    new_dgs = dgs_f / float(rx_dose_gy)

    # Read full dataset (with PixelData)
    try:
        rd_full = pydicom.dcmread(str(rd_path), stop_before_pixels=False, force=True)
    except Exception as e:
        return False, f"Read full RD (with PixelData) failed: {type(e).__name__}: {e}"

    rd_full.DoseGridScaling = new_dgs
    rd_full.DoseUnits = "RELATIVE"

    old_sd = safe_get(rd_full, "SeriesDescription", "")
    suffix = f"Rx-normalized (DoseGridScaling/RxDose; RxSource={rx_source})"
    rd_full.SeriesDescription = (old_sd + " | " + suffix).strip(" |") if old_sd else suffix

    try:
        comment = f"RxDoseGy={rx_dose_gy} (RxSource={rx_source})"
        if n_frac is not None:
            comment += f", FractionsPlanned={n_frac}"
        old_comment = safe_get(rd_full, "DoseComment", "")
        rd_full.DoseComment = (old_comment + " | " + comment).strip(" |") if old_comment else comment
    except Exception:
        pass

    rd_full.SOPInstanceUID = generate_uid()
    rd_full.SeriesInstanceUID = generate_uid()

    now = datetime.now()
    rd_full.InstanceCreationDate = now.strftime("%Y%m%d")
    rd_full.InstanceCreationTime = now.strftime("%H%M%S")

    try:
        rd_full.save_as(str(out_path), write_like_original=False)
    except Exception as e:
        return False, f"Write failed: {type(e).__name__}: {e}"

    return True, f"OK. RxDoseGy={rx_dose_gy} (source={rx_source}), oldDGS={dgs_f}, newDGS={new_dgs}. Notes: {'; '.join(notes)}"


def build_rp_index(patient_dir: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    for rp in patient_dir.rglob("RP*.dcm"):
        try:
            ds = dicom_read(rp)
            if not is_rtplan(ds):
                continue
            uid = str(safe_get(ds, "SOPInstanceUID", "")).strip()
            if uid:
                idx[uid] = rp
        except Exception:
            continue
    return idx


def find_rd_files(patient_dir: Path) -> List[Path]:
    return sorted(patient_dir.rglob("RD*.dcm"))


def patient_has_pt_folder(patient_dir: Path) -> bool:
    key = "_PT_"
    try:
        for p in patient_dir.iterdir():
            if p.is_dir() and key in p.name.upper():
                return True
    except Exception:
        pass

    try:
        for p in patient_dir.rglob("*"):
            if p.is_dir() and key in p.name.upper():
                return True
    except Exception:
        pass

    return False


def append_skip_log(
    log_path: Path,
    reason: str,
    patient: str,
    rd_path: Optional[Path] = None,
    rp_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    detail: Optional[str] = None,
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rd_s = str(rd_path) if rd_path else ""
    rp_s = str(rp_path) if rp_path else ""
    out_s = str(out_path) if out_path else ""
    detail_s = detail if detail else ""
    line = f"{ts}\t{reason}\t{patient}\tRD={rd_s}\tRP={rp_s}\tOUT={out_s}\t{detail_s}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def main():
    root = Path(ROOT_DIR).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root}")

    skip_log_path = root / SKIP_LOG_NAME
    if not skip_log_path.exists():
        with skip_log_path.open("w", encoding="utf-8") as f:
            f.write("timestamp\treason\tpatient\tRD\tRP\tOUT\tdetail\n")

    # NEW: load CSV maxGy map once
    maxgy_map, info = load_patient_maxgy_map(MAXGY_CSV_PATH)
    print(f"[INFO] {info}")

    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Patients: {len(patient_dirs)}")
    print(f"[INFO] Skip log: {skip_log_path}")
    print(f"[INFO] MaxGy CSV: {MAXGY_CSV_PATH}")
    print(f"[INFO] MaxGy reasonable range: [{MAXGY_REASONABLE_MIN}, {MAXGY_REASONABLE_MAX}] Gy")

    n_total_rd = 0
    n_done = 0
    n_skip = 0
    n_exist = 0
    n_skip_no_pt = 0

    for pdir in patient_dirs:
        patient_id = pdir.name

        if not patient_has_pt_folder(pdir):
            msg = "No PT* folder found under patient directory; skipping all RD processing for this patient."
            print(f"[SKIP-PATIENT] {patient_id} | {msg}")
            append_skip_log(
                skip_log_path,
                reason="NoPTFolder",
                patient=patient_id,
                rd_path=None,
                rp_path=None,
                out_path=None,
                detail=msg
            )
            n_skip += 1
            n_skip_no_pt += 1
            continue

        rp_index = build_rp_index(pdir)
        rd_files = find_rd_files(pdir)
        if not rd_files:
            continue

        for rd_path in rd_files:
            n_total_rd += 1
            out_path = rd_path.parent / f"{OUT_PREFIX}{rd_path.name}"

            if out_path.exists():
                print(f"[SKIP] {patient_id} | RD={rd_path.name} | Output already exists: {out_path.name}")
                append_skip_log(
                    skip_log_path,
                    reason="OutputExists",
                    patient=patient_id,
                    rd_path=rd_path,
                    rp_path=None,
                    out_path=out_path,
                    detail="Skipping because output file already exists."
                )
                n_skip += 1
                n_exist += 1
                continue

            try:
                rd_ds = dicom_read(rd_path)
            except Exception as e:
                msg = f"RD read failed: {type(e).__name__}: {e}"
                print(f"[SKIP] {patient_id} {msg} | {rd_path}")
                append_skip_log(
                    skip_log_path,
                    reason="RDReadFailed",
                    patient=patient_id,
                    rd_path=rd_path,
                    rp_path=None,
                    out_path=out_path,
                    detail=msg
                )
                n_skip += 1
                continue

            ref_rp_uid = extract_referenced_rp_uid_from_rd(rd_ds)

            rp_path = None
            if ref_rp_uid and ref_rp_uid in rp_index:
                rp_path = rp_index[ref_rp_uid]
            else:
                candidates = sorted(rd_path.parent.glob("RP*.dcm"))
                if candidates:
                    rp_path = candidates[0]

            if rp_path is None:
                msg = f"No matching RP found (ref_uid={ref_rp_uid})"
                print(f"[SKIP] {patient_id} | RD={rd_path.name} | {msg}")
                append_skip_log(
                    skip_log_path,
                    reason="NoMatchingRP",
                    patient=patient_id,
                    rd_path=rd_path,
                    rp_path=None,
                    out_path=out_path,
                    detail=msg
                )
                n_skip += 1
                continue

            ok, msg = make_rx_normalized_rtdose(
                rd_path=rd_path,
                rp_path=rp_path,
                out_path=out_path,
                patient_id=patient_id,
                maxgy_map=maxgy_map,
            )
            if ok:
                print(f"[DONE] {patient_id} | RD={rd_path.name} -> {out_path.name} | {msg}")
                n_done += 1
            else:
                print(f"[SKIP] {patient_id} | RD={rd_path.name} | {msg}")
                append_skip_log(
                    skip_log_path,
                    reason="ProcessFailed",
                    patient=patient_id,
                    rd_path=rd_path,
                    rp_path=rp_path,
                    out_path=out_path,
                    detail=msg
                )
                n_skip += 1

    print("\n========== SUMMARY ==========")
    print(f"Total RD found:           {n_total_rd}")
    print(f"Generated new RD:         {n_done}")
    print(f"Skipped:                 {n_skip} (includes OutputExists + NoPTFolder + ProcessFailed)")
    print(f"  of which OutputExists:  {n_exist}")
    print(f"  of which NoPTFolder:    {n_skip_no_pt} (patients)")
    print("Output files are written next to original RD with prefix:", OUT_PREFIX)
    print("Skip reasons written to:", skip_log_path)


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from pathlib import Path
# from datetime import datetime

# # =========================
# # CONFIG
# # =========================
# ROOT_DIR = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data"

# # All generated normalized files contain this substring
# TARGET_KEYWORD = "RD_RxNorm"

# # Safety option: if True, only print files (no deletion)
# DRY_RUN = False

# # Log file name
# LOG_NAME = "deleted_RD_RxNorm.txt"


# def main():
#     root = Path(ROOT_DIR).expanduser().resolve()
#     if not root.exists():
#         raise FileNotFoundError(f"ROOT_DIR does not exist: {root}")

#     log_path = root / LOG_NAME

#     # Write log header
#     with log_path.open("w", encoding="utf-8") as f:
#         f.write(f"Delete RD_RxNorm files log\n")
#         f.write(f"Root: {root}\n")
#         f.write(f"Keyword: {TARGET_KEYWORD}\n")
#         f.write(f"DryRun: {DRY_RUN}\n")
#         f.write(f"Time: {datetime.now()}\n")
#         f.write("=" * 100 + "\n\n")

#     # Find all matching files recursively
#     matches = sorted(root.rglob(f"*{TARGET_KEYWORD}*.dcm"))

#     print(f"[INFO] Root folder: {root}")
#     print(f"[INFO] Found {len(matches)} files containing '{TARGET_KEYWORD}'")

#     deleted_count = 0

#     for file_path in matches:
#         print(f"[FOUND] {file_path}")

#         with log_path.open("a", encoding="utf-8") as f:
#             f.write(str(file_path) + "\n")

#         if DRY_RUN:
#             continue

#         try:
#             file_path.unlink()
#             deleted_count += 1
#         except Exception as e:
#             print(f"[ERROR] Could not delete: {file_path} | {e}")
#             with log_path.open("a", encoding="utf-8") as f:
#                 f.write(f"  ERROR: {e}\n")

#     print("\n========== SUMMARY ==========")
#     print(f"Total matched files: {len(matches)}")
#     print(f"Deleted files:       {deleted_count}")
#     print(f"DryRun mode:         {DRY_RUN}")
#     print(f"Log saved to:        {log_path}")

#     if DRY_RUN:
#         print("\n[NOTE] DRY_RUN=True, nothing was deleted.")
#     else:
#         print("\n[OK] Cleanup complete.")


# if __name__ == "__main__":
#     main()
