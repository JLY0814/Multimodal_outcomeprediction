from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicom.uid import generate_uid


# =========================
# CONFIG
# =========================
ROOT = Path(r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data")

CT_REF_DIR = ROOT / "registered_all" / "registered_all_resampled_09766_3mm"

# 输出前缀：加在原 dose 文件名前
OUT_PREFIX = "RD2CT_re_"

# 你说归一化后范围 0~2：可选上限裁剪
CLIP_MIN = 0.0
CLIP_MAX = None  # 如果你不想裁上限，改成 None

# dose resample 插值：连续场推荐线性
DOSE_INTERP = sitk.sitkLinear
DEFAULT_VALUE = 0.0

# 写回 DICOM 时用 uint16 + DoseGridScaling 重新编码
# 选择 scale 使得 [0, CLIP_MAX] 映射到 [0, 65535]
USE_UINT16 = True


# =========================
# Helpers
# =========================
def find_ct_nii(pid: str) -> Path | None:
    p1 = CT_REF_DIR / f"{pid}_CT.nii"
    p2 = CT_REF_DIR / f"{pid}_CT.nii.gz"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return None


def read_rtdose_real_value_as_sitk(dose_path: Path) -> sitk.Image:
    """
    用 pydicom 读 RTDOSE pixel_array，并乘 DoseGridScaling 得到真实值（你这里应为 0~2）。
    再用 sitk.ReadImage 拿空间信息，并 CopyInformation。
    """
    ds = pydicom.dcmread(str(dose_path), force=True)
    dgs = float(getattr(ds, "DoseGridScaling", 1.0))

    raw = ds.pixel_array.astype(np.float32)  # (frames, rows, cols) -> (z,y,x)
    real = raw * dgs

    meta = sitk.ReadImage(str(dose_path))
    img = sitk.GetImageFromArray(real.astype(np.float32))  # (z,y,x)
    img.CopyInformation(meta)
    return img


def resample_to_reference(moving: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """
    将 moving resample 到 reference 的 grid（size/spacing/origin/direction 完全一致）
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())  # identity（假设已在同一物理空间）
    resampler.SetInterpolator(DOSE_INTERP)
    resampler.SetDefaultPixelValue(float(DEFAULT_VALUE))
    out = resampler.Execute(moving)
    return sitk.Cast(out, sitk.sitkFloat32)


def clamp_array(arr: np.ndarray) -> np.ndarray:
    if CLIP_MAX is None:
        return np.clip(arr, CLIP_MIN, None)
    return np.clip(arr, CLIP_MIN, CLIP_MAX)


def ct_direction_to_iop(direction: tuple[float, ...]) -> list[float]:
    """
    SITK direction 是 3x3（按行展平）。
    ITK/SITK：列向量分别是 x/y/z 轴方向。
    DICOM IOP：前3是 row direction（x轴/列方向），后3是 col direction（y轴/行方向）。
    这里用：
      row_dir = x-axis dir = column 0
      col_dir = y-axis dir = column 1
    """
    D = np.array(direction, dtype=np.float64).reshape(3, 3)
    row_dir = D[:, 0]
    col_dir = D[:, 1]
    iop = [float(x) for x in row_dir.tolist() + col_dir.tolist()]
    return iop


def write_resampled_rtdose_dicom(
    original_dose_dcm: Path,
    dose_arr_real: np.ndarray,           # (z,y,x) float32, already in 0~2 and clipped
    ct_ref: sitk.Image,
    out_dcm_path: Path
):
    """
    将 resample 后的 dose 写成一个新的 RTDOSE DICOM：
      - 用 uint16 编码 PixelData
      - 更新 DoseGridScaling
      - 更新 Rows/Columns/NumberOfFrames/GridFrameOffsetVector/PixelSpacing/IOP/IPP 等
    """
    ds = pydicom.dcmread(str(original_dose_dcm), force=True)

    # ---- 准备 grid 参数（来自 ct_ref）----
    # SITK: spacing=(x,y,z), size=(x,y,z)
    sx, sy, sz = ct_ref.GetSpacing()
    nx, ny, nz = ct_ref.GetSize()
    origin = ct_ref.GetOrigin()
    direction = ct_ref.GetDirection()

    # PixelSpacing in DICOM = [row_spacing(dy), col_spacing(dx)]
    pixel_spacing = [float(sy), float(sx)]

    # IOP
    iop = ct_direction_to_iop(direction)

    # GridFrameOffsetVector: offsets along slice direction from first frame
    # 一般用 [0, sz, 2*sz, ...]
    gfov = [float(k * sz) for k in range(nz)]

    # ImagePositionPatient: 使用 ct_ref 的 origin 作为第一帧左上角像素物理坐标（通常可用）
    ipp = [float(origin[0]), float(origin[1]), float(origin[2])]

    # ---- 将真实值编码成 uint16 + scaling ----
    if USE_UINT16:
        # 让 [0, CLIP_MAX] -> [0, 65535]
        # scaling = real_value / stored_value  => stored_value = real_value / scaling
        # 所以 scaling = CLIP_MAX / 65535
        maxv = float(CLIP_MAX) if CLIP_MAX is not None else float(np.max(dose_arr_real))
        if maxv <= 0:
            maxv = 1.0
        scaling = maxv / 65535.0
        stored = np.rint(dose_arr_real / scaling).astype(np.int32)
        stored = np.clip(stored, 0, 65535).astype(np.uint16)
        ds.DoseGridScaling = float(scaling)
        pixel_bytes = stored.tobytes(order="C")
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0  # unsigned
    else:
        # 不推荐：float 写入 DICOM 非标准
        raise ValueError("WRITE_DICOM currently supports uint16 encoding only.")

    # ---- 更新 RTDOSE 关键 tags ----
    ds.Rows = int(ny)
    ds.Columns = int(nx)
    ds.NumberOfFrames = int(nz)

    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    ds.ImageOrientationPatient = [str(x) for x in iop]
    ds.ImagePositionPatient = [str(x) for x in ipp]
    ds.GridFrameOffsetVector = [str(x) for x in gfov]

    # 让一些软件更容易识别为新系列/新实例
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.InstanceNumber = 1

    # 可选：更新描述
    desc = getattr(ds, "SeriesDescription", "")
    new_desc = f"Resampled to CT 0.9766x0.9766x3mm | {desc}".strip()
    ds.SeriesDescription = new_desc[:64]  # 避免太长

    # 写 PixelData
    ds.PixelData = pixel_bytes

    out_dcm_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_dcm_path), write_like_original=False)


def main():
    if not CT_REF_DIR.exists():
        print(f"[ERROR] CT_REF_DIR 不存在: {CT_REF_DIR}")
        sys.exit(1)

    # 找所有 RD_RxNorm*.dcm（文件名包含字段）
    dose_files = sorted([
        p for p in ROOT.rglob("*.dcm")
        if "rd_rxnorm" in p.name.lower() and not p.name.startswith(OUT_PREFIX)
    ])
    print(f"[INFO] Found RD_RxNorm dcm files: {len(dose_files)}")

    # ---- group by parent folder for folder-level skip check ----
    doses_by_folder: dict[Path, list[Path]] = defaultdict(list)
    for dp in dose_files:
        doses_by_folder[dp.parent].append(dp)

    n_ok, n_skip, n_fail = 0, 0, 0

    for dose_folder, folder_doses in sorted(doses_by_folder.items()):

        # ================================================================
        # FOLDER-LEVEL SKIP: if 3+ files with OUT_PREFIX already exist,
        # assume this folder was fully processed and skip it entirely.
        # ================================================================
        existing_results = [
            f for f in dose_folder.iterdir()
            if f.is_file() and f.name.startswith(OUT_PREFIX)
        ]
        if len(existing_results) >= 2:
            print(
                f"[SKIP] {dose_folder.relative_to(ROOT)}: "
                f"already has {len(existing_results)} '{OUT_PREFIX}*' file(s) (expected 2), skipping."
            )
            n_skip += len(folder_doses)
            continue
        # ================================================================

        for dose_path in folder_doses:
            # pid = ROOT 下第一层子目录名：ROOT/GYNDATASET001/...
            try:
                pid = dose_path.relative_to(ROOT).parts[0]
            except Exception:
                print(f"[SKIP] Cannot parse pid: {dose_path}")
                n_skip += 1
                continue

            ct_nii = find_ct_nii(pid)
            if ct_nii is None:
                print(f"[SKIP] {pid}: CT ref nii not found for {dose_path.name}")
                n_skip += 1
                continue

            try:
                # reference CT
                ct_ref = sitk.ReadImage(str(ct_nii))

                # read dose as real value (0~2) with correct scaling
                dose_img = read_rtdose_real_value_as_sitk(dose_path)

                # resample to CT grid
                dose_rs = resample_to_reference(dose_img, ct_ref)

                # to numpy + clip
                dose_arr = sitk.GetArrayFromImage(dose_rs).astype(np.float32)  # (z,y,x)
                dose_arr = clamp_array(dose_arr)

                # 输出文件名：同目录 + 前缀
                out_dir = dose_path.parent
                out_base = out_dir / (OUT_PREFIX + dose_path.stem)

                # 1) save NPY (model input)
                np.save(out_base.with_suffix(".npy"), dose_arr)

                # 2) save resampled DICOM RTDOSE in same folder
                out_dcm = out_base.with_suffix(".dcm")
                write_resampled_rtdose_dicom(
                    original_dose_dcm=dose_path,
                    dose_arr_real=dose_arr,
                    ct_ref=ct_ref,
                    out_dcm_path=out_dcm
                )

                print(f"[OK] {pid}: {dose_path.name} -> {out_base.name}.npy / .dcm  (CT ref={ct_nii.name})")
                n_ok += 1

            except Exception as e:
                print(f"[FAIL] {pid}: {dose_path}  error={repr(e)}")
                n_fail += 1

    print(f"\n[DONE] OK={n_ok}, SKIP={n_skip}, FAIL={n_fail}")


if __name__ == "__main__":
    main()

