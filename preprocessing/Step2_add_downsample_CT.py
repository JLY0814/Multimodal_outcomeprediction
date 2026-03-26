from __future__ import annotations
from pathlib import Path
import math
import shutil
import numpy as np
import pydicom
import SimpleITK as sitk

# =========================
# CONFIG
# =========================
ROOT = Path(r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data")
IN_DIR = ROOT / "registered_all"

OUT_DIR = IN_DIR / "registered_all_resampled_09766_3mm"

OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / "resample_log.txt"

TARGET_SPACING = (0.9766, 0.9766, 3.0)  # (x,y,z) mm
TOL = 0.01  # per-axis tolerance

# Patients with CT pixel spacing < 0.9766 that need anti-aliasing
HIGH_RES_CT_PATIENTS = {
    'GYNDATASET070', 'GYNDATASET074', 'GYNDATASET086', 'GYNDATASET095',
    'GYNDATASET104', 'GYNDATASET121', 'GYNDATASET170', 'GYNDATASET171',
    'GYNDATASET178', 'GYNDATASET192', 'GYNDATASET196', 'GYNDATASET200',
    'GYNDATASET212', 'GYNDATASET231', 'GYNDATASET247', 'GYNDATASET262',
    'GYNDATASET275', 'GYNDATASET289', 'GYNDATASET312', 'GYNDATASET322',
    'GYNDATASET345', 'GYNDATASET348', 'GYNDATASET365', 'GYNDATASET366',
    'GYNDATASET377', 'GYNDATASET420'
}


# =========================
# Logging
# =========================
def log(msg: str):
    print(msg)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


# =========================
# Utilities
# =========================
def patient_id_from_nii_name(name: str) -> str | None:
    # expected: GYNDATASET298_CT.nii or GYNDATASET298_PET.nii.gz
    base = name
    if base.lower().endswith(".nii.gz"):
        base = base[:-7]
    elif base.lower().endswith(".nii"):
        base = base[:-4]
    else:
        return None

    if "_CT" in base:
        return base.split("_CT")[0]
    if "_PET" in base:
        return base.split("_PET")[0]
    return None


def is_ct_image(nii_name: str) -> bool:
    """Check if this is a CT image based on filename."""
    return "_CT" in nii_name


def npy_path_for_nii(nii_path: Path) -> Path:
    # keep same stem mapping as in your folder: xxx_CT.npy or xxx_PET.npy
    name = nii_path.name
    if name.lower().endswith(".nii.gz"):
        base = name[:-7]
    elif name.lower().endswith(".nii"):
        base = name[:-4]
    else:
        base = nii_path.stem
    return nii_path.with_name(base + ".npy")


def spacing_is_unreliable(sp) -> bool:
    """Heuristic: missing/garbage spacing in NIfTI often shows up as (1,1,1), 0, NaN, etc."""
    if sp is None or len(sp) < 3:
        return True
    x, y, z = sp[:3]
    vals = [x, y, z]
    if any((v is None) or (not math.isfinite(v)) for v in vals):
        return True
    if any(v <= 0 for v in vals):
        return True
    # suspicious all ones
    if abs(x - 1.0) < 1e-6 and abs(y - 1.0) < 1e-6 and abs(z - 1.0) < 1e-6:
        return True
    # absurd ranges
    if not (0.1 <= x <= 10 and 0.1 <= y <= 10 and 0.1 <= z <= 30):
        return True
    return False


def spacing_close(sp, target, tol=TOL) -> bool:
    return all(abs(float(sp[i]) - float(target[i])) < tol for i in range(3))


def needs_downsampling(old_spacing, new_spacing) -> tuple[bool, list[int]]:
    """
    Check if any axis requires downsampling (old < new).
    Returns (needs_downsampling, list of axis indices that need downsampling)
    """
    downsample_axes = []
    for i in range(3):
        if old_spacing[i] < new_spacing[i]:
            downsample_axes.append(i)
    return len(downsample_axes) > 0, downsample_axes


def apply_antialiasing_filter(img: sitk.Image, old_spacing, new_spacing) -> sitk.Image:
    """
    Apply Gaussian smoothing to prevent aliasing when downsampling.
    Uses a sigma based on the downsampling ratio for each axis.
    
    Theory: When downsampling, we need to low-pass filter to remove frequencies
    above the Nyquist limit of the new sampling rate. A Gaussian with sigma
    proportional to the downsampling ratio achieves this.
    """
    sigma_physical = []  # in mm (physical space)
    
    for i in range(3):
        if new_spacing[i] > old_spacing[i]:
            # Downsampling: need anti-aliasing
            # Sigma in physical units (mm)
            # Rule of thumb: sigma ≈ 0.5 * new_spacing works well
            # This corresponds to smoothing to the Nyquist frequency of target resolution
            sigma_physical.append(0.5 * new_spacing[i])
        else:
            # Upsampling or no change: no smoothing needed
            sigma_physical.append(0.0)
    
    if any(s > 0 for s in sigma_physical):
        log(f"      Anti-aliasing: applying Gaussian filter with sigma={sigma_physical} mm")
        # SmoothingRecursiveGaussian uses physical units (mm)
        img_smoothed = sitk.SmoothingRecursiveGaussian(img, sigma_physical)
        return img_smoothed
    else:
        return img


def resample_sitk(img: sitk.Image, out_spacing, is_label=False, apply_antialiasing=False) -> sitk.Image:
    """
    Resample image to new spacing with optional anti-aliasing.
    
    Args:
        img: Input SimpleITK image
        out_spacing: Target spacing (x, y, z) in mm
        is_label: If True, use nearest neighbor interpolation
        apply_antialiasing: If True, apply Gaussian smoothing before downsampling
    """
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()

    # Calculate new size
    new_size = [
        int(round(old_size[i] * old_spacing[i] / out_spacing[i]))
        for i in range(3)
    ]
    new_size = [max(1, s) for s in new_size]

    # Apply anti-aliasing filter if needed and requested
    img_to_resample = img
    if apply_antialiasing and not is_label:
        needs_aa, downsample_axes = needs_downsampling(old_spacing, out_spacing)
        if needs_aa:
            img_to_resample = apply_antialiasing_filter(img, old_spacing, out_spacing)

    # Setup resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(out_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Use linear interpolation (fast and good quality)
        # Alternative: sitk.sitkBSpline for higher quality (slower)
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img_to_resample)


def find_rp_matched_dir(patient_dir: Path) -> Path | None:
    # find any folder containing "RP_matched" (case-insensitive)
    for d in patient_dir.rglob("*"):
        if d.is_dir() and "rp_matched" in d.name.lower():
            return d
    return None


def pick_one_series_dicoms(dcm_root: Path, max_files=8000) -> list[Path]:
    """
    Collect dicoms under dcm_root. If mixed series exist, pick the largest consistent SeriesInstanceUID.
    """
    # DICOMs might have .dcm or no extension
    all_files = [p for p in dcm_root.rglob("*") if p.is_file()]
    if len(all_files) > max_files:
        all_files = all_files[:max_files]

    series_map: dict[str, list[Path]] = {}
    for p in all_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            if not uid:
                continue
            series_map.setdefault(uid, []).append(p)
        except Exception:
            continue

    if not series_map:
        return []
    best_uid = max(series_map.keys(), key=lambda k: len(series_map[k]))
    return series_map[best_uid]


def estimate_ct_spacing_from_dicoms(dcm_dir: Path) -> tuple[float, float, float] | None:
    """
    Return (sx, sy, sz) from CT dicoms.
    - sx, sy from PixelSpacing
    - sz from median delta of ImagePositionPatient along slice normal (robust),
      fallback to SpacingBetweenSlices or SliceThickness.
    """
    series_files = pick_one_series_dicoms(dcm_dir)
    if not series_files:
        return None

    rep = None
    for p in series_files[:100]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            if hasattr(ds, "PixelSpacing"):
                rep = ds
                break
        except Exception:
            continue
    if rep is None or not hasattr(rep, "PixelSpacing"):
        return None

    sy, sx = float(rep.PixelSpacing[0]), float(rep.PixelSpacing[1])

    normals = None
    if hasattr(rep, "ImageOrientationPatient"):
        iop = [float(x) for x in rep.ImageOrientationPatient]
        row = np.array(iop[:3], dtype=np.float64)
        col = np.array(iop[3:], dtype=np.float64)
        n = np.cross(row, col)
        norm = np.linalg.norm(n)
        if norm > 0:
            normals = n / norm

    positions = []
    for p in series_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            if hasattr(ds, "ImagePositionPatient"):
                ipp = np.array([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)
                if normals is not None:
                    positions.append(float(np.dot(ipp, normals)))
                else:
                    positions.append(float(ipp[2]))
        except Exception:
            continue

    sz = None
    if len(positions) >= 2:
        positions = np.array(sorted(positions), dtype=np.float64)
        diffs = np.abs(np.diff(positions))
        diffs = diffs[diffs > 1e-6]
        if diffs.size > 0:
            sz = float(np.median(diffs))

    if sz is None:
        if hasattr(rep, "SpacingBetweenSlices"):
            sz = float(rep.SpacingBetweenSlices)
        elif hasattr(rep, "SliceThickness"):
            sz = float(rep.SliceThickness)

    if sz is None or (not math.isfinite(sz)) or sz <= 0:
        return None

    return (sx, sy, sz)


def load_dicom_series_as_nifti(dcm_dir: Path) -> sitk.Image | None:
    """
    Load a DICOM series from a directory and return as SimpleITK Image.
    """
    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dcm_dir))
        
        if not dicom_names:
            return None
        
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        
        img = reader.Execute()
        return img
    except Exception as e:
        log(f"[ERR ] Failed to load DICOM series from {dcm_dir}: {repr(e)}")
        return None


# =========================
# MAIN PROCESSING
# =========================
log(f"ROOT={ROOT}")
log(f"IN_DIR={IN_DIR}")
log(f"OUT_DIR={OUT_DIR}")
log(f"TARGET_SPACING={TARGET_SPACING}, TOL={TOL}")
log(f"High-resolution CT patients requiring anti-aliasing: {len(HIGH_RES_CT_PATIENTS)}")

# =========================
# 1) Scan existing NIfTI files to determine which patients need processing
# =========================
log("---- Scanning NIfTI files in registered_all ----")

nii_paths = sorted(list(IN_DIR.glob("*.nii")) + list(IN_DIR.glob("*.nii.gz")))
log(f"Found {len(nii_paths)} NIfTI files")

# Build set of patient IDs that have NIfTI files
patients_with_nifti = set()
for nii_path in nii_paths:
    pid = patient_id_from_nii_name(nii_path.name)
    if pid:
        patients_with_nifti.add(pid)

log(f"Unique patients with NIfTI files: {len(patients_with_nifti)}")

# =========================
# 2) Build CT spacing map ONLY for patients that need processing
# =========================
log("---- Building CT spacing map for required patients only ----")

ct_spacing_map: dict[str, tuple[float, float, float]] = {}
patient_dicom_dirs: dict[str, Path] = {}  # Store DICOM dirs for regeneration

for patient_dir in ROOT.iterdir():
    if not patient_dir.is_dir():
        continue
    if patient_dir.name.lower() == "registered_all":
        continue
    
    # Skip patients not in our processing list
    if patient_dir.name not in patients_with_nifti:
        continue

    rp_dir = find_rp_matched_dir(patient_dir)
    if rp_dir is None:
        log(f"[WARN] {patient_dir.name}: No RP_matched folder found, cannot get spacing")
        continue

    sp = estimate_ct_spacing_from_dicoms(rp_dir)
    if sp is None:
        log(f"[WARN] {patient_dir.name}: RP_matched found but failed to estimate spacing ({rp_dir})")
        continue

    ct_spacing_map[patient_dir.name] = sp
    patient_dicom_dirs[patient_dir.name] = rp_dir
    
    # Log if this is a high-resolution CT
    if patient_dir.name in HIGH_RES_CT_PATIENTS:
        log(f"[OK-HR] {patient_dir.name}: CT spacing={sp} [HIGH-RES]")
    else:
        log(f"[OK]    {patient_dir.name}: CT spacing={sp}")

log(f"---- Processed {len(ct_spacing_map)} patients with CT spacing ----")


# =========================
# 3) Process NIfTI in registered_all (with regeneration if needed)
# =========================
log("---- Processing NIfTI files ----")

for nii_path in nii_paths:
    pid = patient_id_from_nii_name(nii_path.name)
    if pid is None:
        log(f"[SKIP] {nii_path.name}: cannot parse patient id")
        continue

    # Check if NIfTI file actually exists (might be in list but deleted)
    if not nii_path.exists():
        log(f"[MISSING] {nii_path.name}: file does not exist, attempting to regenerate from DICOM")
        
        # Try to regenerate from DICOM
        if pid in patient_dicom_dirs:
            dcm_dir = patient_dicom_dirs[pid]
            img = load_dicom_series_as_nifti(dcm_dir)
            
            if img is not None:
                # Save to IN_DIR first
                sitk.WriteImage(img, str(nii_path))
                log(f"[REGEN] {nii_path.name}: regenerated from DICOM at {dcm_dir.name}")
            else:
                log(f"[ERR ] {nii_path.name}: failed to regenerate from DICOM")
                continue
        else:
            log(f"[ERR ] {nii_path.name}: no DICOM directory available for regeneration")
            continue

    if pid not in ct_spacing_map:
        log(f"[SKIP] {nii_path.name}: no CT dicom spacing for patient {pid}")
        continue

    ct_sp = ct_spacing_map[pid]
    is_ct = is_ct_image(nii_path.name)
    
    # Determine if anti-aliasing is needed
    # Apply anti-aliasing for CT images from high-resolution patients
    use_antialiasing = is_ct and (pid in HIGH_RES_CT_PATIENTS)

    out_nii = OUT_DIR / nii_path.name
    in_npy = npy_path_for_nii(nii_path)
    out_npy = OUT_DIR / in_npy.name

    try:
        img = sitk.ReadImage(str(nii_path))
        sp0 = img.GetSpacing()

        # Step A: fix unreliable spacing using CT dicom spacing (anchor)
        if spacing_is_unreliable(sp0):
            img.SetSpacing(tuple(ct_sp))
            log(f"[FIX ] {nii_path.name}: spacing {sp0} -> set to CT spacing {ct_sp}")
            sp0 = img.GetSpacing()

        # Step B: decide whether to resample
        if spacing_close(sp0, TARGET_SPACING, TOL):
            # spacing already close enough: no resample
            shutil.copy2(nii_path, out_nii)
            
            if use_antialiasing:
                log(f"[KEEP] {nii_path.name}: spacing={sp0} close to target, copied NIfTI [HIGH-RES CT, no resample needed]")
            else:
                log(f"[KEEP] {nii_path.name}: spacing={sp0} close to target, copied NIfTI")

            # for NPY: prefer copying existing if exists; otherwise generate from image
            if in_npy.exists():
                shutil.copy2(in_npy, out_npy)
                log(f"[KEEP] {in_npy.name}: copied existing NPY")
            else:
                arr = sitk.GetArrayFromImage(img)  # (z,y,x)
                np.save(out_npy, arr)
                log(f"[MAKE] {out_npy.name}: generated NPY from image (no input NPY)")

        else:
            # need resample to target spacing
            if use_antialiasing:
                log(f"[RSMP-AA] {nii_path.name}: {sp0} -> {TARGET_SPACING} [HIGH-RES CT with anti-aliasing]")
            else:
                log(f"[RSMP] {nii_path.name}: {sp0} -> {TARGET_SPACING}")
            
            out_img = resample_sitk(img, TARGET_SPACING, is_label=False, apply_antialiasing=use_antialiasing)
            sitk.WriteImage(out_img, str(out_nii))
            
            if use_antialiasing:
                log(f"[RSMP-AA] wrote {out_nii.name} with anti-aliasing filter applied")
            else:
                log(f"[RSMP] wrote {out_nii.name}")

            # always regenerate NPY to match the new NIfTI
            arr = sitk.GetArrayFromImage(out_img)  # (z,y,x)
            np.save(out_npy, arr)
            log(f"[RSMP] {out_npy.name}: regenerated NPY from resampled image")

        # Step C: delete old NPY in IN_DIR (only after output NPY exists)
        if out_npy.exists() and in_npy.exists():
            try:
                in_npy.unlink()
                log(f"[DEL ] deleted old NPY: {in_npy.name}")
            except Exception as e_del:
                log(f"[WARN] failed to delete old NPY {in_npy.name}: {repr(e_del)}")

    except Exception as e:
        log(f"[ERR ] {nii_path.name}: {repr(e)}")

log("---- All done ----")
log(f"Output folder: {OUT_DIR}")
log(f"Log file: {LOG_PATH}")
log(f"Processed {len(patients_with_nifti)} unique patients")
log(f"Anti-aliasing was applied to CT images from {len(HIGH_RES_CT_PATIENTS)} high-resolution patients")