from __future__ import annotations

from pathlib import Path
import numpy as np

# 建议装 SimpleITK：conda install -c simpleitk simpleitk  或 pip install SimpleITK
import SimpleITK as sitk
import pydicom


# ==============
# 你需要改的路径
# ==============
ROOT_DIR = Path(r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data")

# 输出目录（不存在会创建）
OUT_DIR = ROOT_DIR / "dose_npy_resampled_to_ct"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 搜索规则
RD_PREFIX = "RD_RxNorm"          # dose 文件名以这个开头
RP_MATCHED_KEY = "RP_matched"    # CT dcm 所在子文件夹名包含这个字段


def is_dicom_file(p: Path) -> bool:
    # 不要太严格：有些 dcm 没后缀
    return p.is_file() and (p.suffix.lower() in ["", ".dcm"])


def find_rd_file(patient_dir: Path) -> Path | None:
    # 递归找 RD_RxNorm*.dcm（或无后缀）
    for p in patient_dir.rglob("*"):
        if p.is_file() and p.name.startswith(RD_PREFIX):
            return p
    return None


def find_rp_matched_ct_folder(patient_dir: Path) -> Path | None:
    # 找包含 RP_matched 的文件夹，且里面确实有 dicom 文件
    candidates = [d for d in patient_dir.rglob("*") if d.is_dir() and RP_MATCHED_KEY in d.name]
    for d in candidates:
        # 这里假设 RP_matched 下面就是 CT 的 dcm（或者再往下也行）
        dicoms = [p for p in d.rglob("*") if is_dicom_file(p)]
        if len(dicoms) >= 5:  # 随便设个阈值，避免误判
            return d
    return None


def read_ct_series_as_reference(ct_folder: Path) -> sitk.Image:
    """
    用 SimpleITK 读 CT series，返回 reference image（包含 size/spacing/origin/direction）
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_folder))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under: {ct_folder}")

    # 如果有多个 series，默认取第一个（通常 RP_matched 里就一个）
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(str(ct_folder), series_id)
    reader.SetFileNames(file_names)

    ct_img = reader.Execute()  # sitk.Image
    return ct_img


def read_dose(rd_path: Path) -> sitk.Image:
    """
    读 RTDOSE DICOM。
    注意：SimpleITK 读取 RTDOSE 的时候会处理 DoseGridScaling（多数情况下）
    """
    return sitk.ReadImage(str(rd_path))


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, is_label: bool = False) -> sitk.Image:
    """
    把 moving 重采样到 reference 网格。
    RTDOSE 是连续值 => 用 Linear 插值更合适（is_label=False）
    """
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    default_value = 0.0

    resampled = sitk.Resample(
        moving,
        reference,
        sitk.Transform(),   # identity
        interpolator,
        default_value,
        moving.GetPixelID()
    )
    return resampled


def patient_id_from_path(patient_dir: Path) -> str:
    # 如果你病人文件夹名就是 GYNDATASETxxx，就直接用它
    return patient_dir.name


def main():
    # 你可以按实际结构调整：如果 ROOT_DIR 下就是很多 GYNDATASETxxx 文件夹，这样最合适
    patient_dirs = [d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("GYNDATASET")]
    print(f"Found {len(patient_dirs)} patient folders under ROOT_DIR.")

    n_ok = 0
    n_skip = 0
    n_err = 0

    for pdir in sorted(patient_dirs):
        pid = patient_id_from_path(pdir)

        try:
            rd_path = find_rd_file(pdir)
            if rd_path is None:
                print(f"[SKIP] {pid}: no {RD_PREFIX}* dose file found")
                n_skip += 1
                continue

            ct_folder = find_rp_matched_ct_folder(pdir)
            if ct_folder is None:
                print(f"[SKIP] {pid}: no *{RP_MATCHED_KEY}* CT folder found")
                n_skip += 1
                continue

            ct_ref = read_ct_series_as_reference(ct_folder)
            dose_img = read_dose(rd_path)

            print("Dose pixel type:", dose_img.GetPixelIDTypeAsString())
            tmp = sitk.GetArrayFromImage(dose_img)
            print("Dose raw min/max:", float(tmp.min()), float(tmp.max()))
            
            ds = pydicom.dcmread(str(rd_path), stop_before_pixels=True, force=True)
            print("DoseGridScaling:", float(getattr(ds, "DoseGridScaling", 1.0)))



            dose_on_ct = resample_to_reference(dose_img, ct_ref, is_label=False)

            # sitk -> numpy (z, y, x)
            dose_arr = sitk.GetArrayFromImage(dose_on_ct)

            # sanity: CT shape
            ct_arr = sitk.GetArrayFromImage(ct_ref)
            if dose_arr.shape != ct_arr.shape:
                # 理论上 resample 后应该一致
                print(f"[WARN] {pid}: dose shape {dose_arr.shape} != CT shape {ct_arr.shape} (still saving)")
            else:
                print(f"[OK] {pid}: dose shape matches CT: {dose_arr.shape}")

            out_path = OUT_DIR / f"{pid}_RxDose.npy"
            np.save(str(out_path), dose_arr.astype(np.float32, copy=False))

            n_ok += 1

        except Exception as e:
            print(f"[ERROR] {pid}: {type(e).__name__}: {e}")
            n_err += 1

    print("\nDone.")
    print(f"OK: {n_ok}, SKIP: {n_skip}, ERROR: {n_err}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
