import os
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path

#######################
#this code will regitered PET to CT use inverse matrix from RE file, and resampling PET to CT with same pixel spacing and thickness.
# the RE file is export from epic.
#######################

# Path containing the reference .npy files used for filtering patients
NPY_FILTER_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all\test_all\test_all_PET\normalized_PET_invalid_mean"


def get_patient_ids_from_npy_folder(npy_folder):
    """
    Scan the npy_folder for *_PET.npy files and extract the patient ID
    portion (everything before '_PET.npy').

    Example:
        GYNDATASET017_PET.npy  ->  GYNDATASET017
        GYNDATASET011_PET.npy  ->  GYNDATASET011

    Returns:
        set of patient ID strings, e.g. {'GYNDATASET017', 'GYNDATASET011', ...}
    """
    if not os.path.isdir(npy_folder):
        raise FileNotFoundError(
            f"NPY filter folder does not exist: {npy_folder}"
        )

    patient_ids = set()
    for fname in os.listdir(npy_folder):
        if fname.endswith("_PET.npy"):
            # Strip '_PET.npy' to get the patient ID
            patient_id = fname[: -len("_PET.npy")]
            if patient_id:
                patient_ids.add(patient_id)

    if not patient_ids:
        raise ValueError(
            f"No *_PET.npy files found in NPY filter folder: {npy_folder}"
        )

    return patient_ids


def load_dicom_series(dicom_folder):
    """
    加载 DICOM 系列
    改进版：更灵活地处理 PET 和 CT DICOM 文件

    Returns:
        image: SimpleITK Image
    """
    reader = sitk.ImageSeriesReader()
    
    # 方法1: 尝试使用 GetGDCMSeriesFileNames (标准方法)
    try:
        series_IDs = reader.GetGDCMSeriesIDs(dicom_folder)
        if series_IDs:
            # 如果找到多个系列，使用第一个
            series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_IDs[0])
            if series_file_names:
                reader.SetFileNames(series_file_names)
                image = reader.Execute()
                return image
    except Exception as e:
        print(f"    Warning: GetGDCMSeriesFileNames failed: {e}")
    
    # 方法2: 如果方法1失败，直接读取所有 .dcm 文件
    print(f"    Trying alternative method: reading all .dcm files...")
    dcm_files = []
    for root, dirs, files in os.walk(dicom_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                dcm_files.append(os.path.join(root, file))
    
    if len(dcm_files) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    # 按文件名排序
    dcm_files.sort()
    
    print(f"    Found {len(dcm_files)} DICOM files")
    
    # 读取第一个文件检查是否是图像
    try:
        ds = pydicom.dcmread(dcm_files[0])
        if not hasattr(ds, 'PixelData'):
            # 如果第一个文件不包含图像数据，过滤掉非图像文件
            print(f"    Filtering non-image DICOM files...")
            image_files = []
            for f in dcm_files:
                ds_temp = pydicom.dcmread(f)
                if hasattr(ds_temp, 'PixelData'):
                    image_files.append(f)
            dcm_files = image_files
            print(f"    After filtering: {len(dcm_files)} image files")
    except Exception as e:
        print(f"    Warning during file filtering: {e}")
    
    if len(dcm_files) == 0:
        raise ValueError(f"No valid DICOM image files found in {dicom_folder}")
    
    # 设置文件名并读取
    reader.SetFileNames(dcm_files)
    image = reader.Execute()
    
    return image


def load_dicom_reg(reg_file):
    """
    加载 DICOM REG 配准文件，提取变换矩阵
    使用 (0x3006, 0x00C6) 标签并跳过单位矩阵

    Returns:
        matrix: 4x4 numpy array (如果找到非单位矩阵)
    """
    ds = pydicom.dcmread(reg_file)

    # 检查是否是 REG 文件
    if ds.Modality != 'REG':
        print(f"Warning: File modality is {ds.Modality}, not REG")

    # 尝试提取刚性/仿射变换
    if hasattr(ds, 'RegistrationSequence'):
        reg_seq = ds.RegistrationSequence

        for reg_item in reg_seq:
            if hasattr(reg_item, 'MatrixRegistrationSequence'):
                matrix_seq = reg_item.MatrixRegistrationSequence

                for matrix_item in matrix_seq:
                    if hasattr(matrix_item, 'MatrixSequence'):
                        for mat_seq in matrix_item.MatrixSequence:
                            # 使用特定的 DICOM 标签 (0x3006, 0x00C6)
                            if (0x3006, 0x00C6) in mat_seq:
                                raw_matrix = mat_seq[(0x3006, 0x00C6)].value
                                parts = [float(x) for x in raw_matrix]
                                
                                if len(parts) == 16:
                                    matrix = np.array(parts, dtype=np.float64).reshape((4, 4))
                                    
                                    print("Found transformation matrix:")
                                    print(matrix)
                                    
                                    # 跳过单位矩阵
                                    if not np.allclose(matrix, np.eye(4)):
                                        print("✓ Non-identity matrix found")
                                        return matrix
                                    else:
                                        print("⚠ Skipping identity matrix")

    # 如果没有找到有效矩阵
    print("Could not extract valid non-identity transform from REG file.")
    print("\nREG file structure:")
    print(f"  Modality: {ds.Modality}")
    if hasattr(ds, 'RegistrationSequence'):
        print(f"  Has RegistrationSequence: Yes ({len(ds.RegistrationSequence)} items)")
    
    return None


def apply_registration_sitk(fixed_image, moving_image, matrix4x4, inverse=False):
    """
    使用 SimpleITK 应用配准变换

    Args:
        fixed_image: 参考图像 (CT)
        moving_image: 要变换的图像 (PET)
        matrix4x4: 4x4 numpy array
        inverse: 是否使用逆变换

    Returns:
        registered_image: 配准后的图像
    """
    if matrix4x4.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {matrix4x4.shape}")

    print("\n" + "="*60)
    print("TRANSFORM DIAGNOSTICS")
    print("="*60)
    
    print("\nOriginal 4x4 Matrix:")
    print(matrix4x4)
    
    # 如果需要逆变换
    if inverse:
        print("\n>>> INVERTING TRANSFORM (PET->CT) <<<")
        matrix4x4 = np.linalg.inv(matrix4x4)
        print("Inverse 4x4 Matrix:")
        print(matrix4x4)
    else:
        print("\n>>> Using ORIGINAL transform (CT->PET) <<<")
    
    # 创建 SimpleITK AffineTransform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(matrix4x4[0:3, 0:3].flatten().tolist())
    transform.SetTranslation(tuple(matrix4x4[0:3, 3].tolist()))
    
    print(f"\nSimpleITK Transform:")
    print(f"  Matrix: {transform.GetMatrix()}")
    print(f"  Translation: {transform.GetTranslation()}")
    
    # 打印图像信息
    print("\nImage Information:")
    print(f"  Fixed (CT) - Size: {fixed_image.GetSize()}, Spacing: {fixed_image.GetSpacing()}")
    print(f"  Moving (PET) - Size: {moving_image.GetSize()}, Spacing: {moving_image.GetSpacing()}")
    
    # 应用变换 - 使用 -1024 作为默认值（CT的空气值）
    print("\nApplying transform...")
    resampled = sitk.Resample(
        moving_image,
        fixed_image,           # 输出网格 = fixed image
        transform,
        sitk.sitkLinear,
        -1024.0,              # CT的空气值作为背景
        moving_image.GetPixelID()
    )
    
    # 检查结果
    registered_array = sitk.GetArrayFromImage(resampled)
    moving_array = sitk.GetArrayFromImage(moving_image)
    
    print("\nTransform Result:")
    print(f"  Original PET - min: {moving_array.min():.2f}, max: {moving_array.max():.2f}, mean: {moving_array.mean():.2f}")
    print(f"  Registered PET - min: {registered_array.min():.2f}, max: {registered_array.max():.2f}, mean: {registered_array.mean():.2f}")
    print(f"  Non-zero voxels: {np.count_nonzero(registered_array > -1000)} / {registered_array.size}")
    
    if registered_array.max() <= 0:
        print("\n⚠ WARNING: Registered image appears empty (all values <= 0)!")
    else:
        print("\n✓ Transform appears successful")
    
    print("="*60 + "\n")

    return resampled


def resample_pet_to_ct_space(ct_image, pet_image):
    """
    简单地将 PET 重采样到 CT 空间 (不使用额外变换)
    假设它们已经在同一坐标系中，只是分辨率不同

    Args:
        ct_image: SimpleITK CT image (参考)
        pet_image: SimpleITK PET image (要重采样)

    Returns:
        resampled_pet: 重采样后的 PET
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    # 使用 identity transform
    resampler.SetTransform(sitk.Transform())

    resampled = resampler.Execute(pet_image)

    return resampled


def save_as_npy_and_nii(ct_image, pet_registered, output_folder, patient_id):
    """
    保存图像为 NPY 和 NIfTI 格式

    Args:
        ct_image: SimpleITK CT image
        pet_registered: SimpleITK registered PET image
        output_folder: 输出文件夹
        patient_id: 患者 ID

    Returns:
        (ct_npy_path, pet_npy_path, ct_nii_path, pet_nii_path)
    """
    # 转换为 numpy 数组
    ct_array = sitk.GetArrayFromImage(ct_image)
    pet_array = sitk.GetArrayFromImage(pet_registered)

    # ========== 保存 NPY 文件 ==========
    ct_npy_path = os.path.join(output_folder, f"{patient_id}_CT.npy")
    pet_npy_path = os.path.join(output_folder, f"{patient_id}_PET.npy")

    np.save(ct_npy_path, ct_array.astype(np.float32))
    np.save(pet_npy_path, pet_array.astype(np.float32))

    print(f"Saved NPY files:")
    print(f"  {ct_npy_path}")
    print(f"  {pet_npy_path}")

    # ========== 保存 NIfTI 文件 ==========
    ct_nii_path = os.path.join(output_folder, f"{patient_id}_CT.nii.gz")
    pet_nii_path = os.path.join(output_folder, f"{patient_id}_PET.nii.gz")

    sitk.WriteImage(ct_image, ct_nii_path)
    sitk.WriteImage(pet_registered, pet_nii_path)

    print(f"Saved NIfTI files:")
    print(f"  {ct_nii_path}")
    print(f"  {pet_nii_path}")

    return ct_npy_path, pet_npy_path, ct_nii_path, pet_nii_path


def get_for_uid(dicom_folder):
    """
    从 DICOM 文件夹中读取第一个 .dcm 文件的 FrameOfReferenceUID。

    Returns:
        str | None: FrameOfReferenceUID 的值，如果无法读取则返回 None
    """
    for fname in sorted(os.listdir(dicom_folder)):
        if not fname.lower().endswith('.dcm'):
            continue
        filepath = os.path.join(dicom_folder, fname)
        if not os.path.isfile(filepath):
            continue
        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            if hasattr(ds, 'FrameOfReferenceUID'):
                return str(ds.FrameOfReferenceUID)
        except Exception as e:
            print(f"    Warning: could not read FrameOfReferenceUID from {fname}: {e}")
            continue
    return None


def process_single_patient(
    ct_folder,
    pet_folder,
    reg_file,
    output_folder,
    patient_id,
    method='resample_only',
    use_inverse_transform=True,
    try_both_directions=False
):
    """
    处理单个患者

    Args:
        ct_folder: CT DICOM 文件夹
        pet_folder: PET DICOM 文件夹
        reg_file: REG 配准文件路径 (可以是 None)
        output_folder: 输出文件夹
        patient_id: 患者 ID
        method: 处理方法
        use_inverse_transform: 如果REG文件是CT->PET,设为True获取PET->CT的逆变换
        try_both_directions: 如果True,会尝试正向和逆向两种变换并选择最好的
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")

    # 1. 加载 CT
    print("Loading CT...")
    ct_image = load_dicom_series(ct_folder)
    print(f"  CT size: {ct_image.GetSize()}")
    print(f"  CT spacing: {ct_image.GetSpacing()}")
    print(f"  CT origin: {ct_image.GetOrigin()}")

    # 2. 加载 PET
    print("Loading PET...")
    pet_image = load_dicom_series(pet_folder)
    print(f"  PET size: {pet_image.GetSize()}")
    print(f"  PET spacing: {pet_image.GetSpacing()}")
    print(f"  PET origin: {pet_image.GetOrigin()}")

    # 3. 比较 FrameOfReferenceUID —— 如果相同则直接 identity resample，忽略 REG
    ct_for_uid  = get_for_uid(ct_folder)
    pet_for_uid = get_for_uid(pet_folder)
    print(f"\n  FrameOfReferenceUID check:")
    print(f"    CT :  {ct_for_uid}")
    print(f"    PET:  {pet_for_uid}")

    same_for = (ct_for_uid is not None
                and pet_for_uid is not None
                and ct_for_uid == pet_for_uid)

    if same_for:
        print("  ✓ FoR UIDs MATCH → using identity resample (REG ignored)")
        pet_registered = resample_pet_to_ct_space(ct_image, pet_image)

    # 4. 应用配准（仅当 FoR UID 不同时执行）
    elif method == 'apply_transform' and reg_file and os.path.exists(reg_file):
        print("Loading REG file...")
        matrix4x4 = load_dicom_reg(reg_file)

        if matrix4x4 is not None:
            if try_both_directions:
                print("\n" + "="*60)
                print("TRYING BOTH TRANSFORM DIRECTIONS")
                print("="*60)
                
                # 尝试正向变换
                print("\n### Testing FORWARD transform (original) ###")
                pet_forward = apply_registration_sitk(ct_image, pet_image, matrix4x4, inverse=False)
                forward_array = sitk.GetArrayFromImage(pet_forward)
                forward_score = np.count_nonzero(forward_array > -1000)
                
                # 尝试逆向变换
                print("\n### Testing INVERSE transform ###")
                pet_inverse = apply_registration_sitk(ct_image, pet_image, matrix4x4, inverse=True)
                inverse_array = sitk.GetArrayFromImage(pet_inverse)
                inverse_score = np.count_nonzero(inverse_array > -1000)
                
                # 选择更好的结果
                print("\n" + "="*60)
                print("COMPARISON")
                print("="*60)
                print(f"Forward transform - Non-zero voxels (>-1000): {forward_score}")
                print(f"Inverse transform - Non-zero voxels (>-1000): {inverse_score}")
                
                if inverse_score > forward_score:
                    print("\n✓ Using INVERSE transform (better result)")
                    pet_registered = pet_inverse
                else:
                    print("\n✓ Using FORWARD transform (better result)")
                    pet_registered = pet_forward
                print("="*60)
            else:
                print("Applying registration transform...")
                pet_registered = apply_registration_sitk(
                    ct_image, 
                    pet_image, 
                    matrix4x4,
                    inverse=use_inverse_transform
                )
        else:
            print("Could not load transform, using resample only...")
            pet_registered = resample_pet_to_ct_space(ct_image, pet_image)
    else:
        print("Resampling PET to CT space...")
        pet_registered = resample_pet_to_ct_space(ct_image, pet_image)

    print(f"  Registered PET size: {pet_registered.GetSize()}")

    # 5. 转换为 numpy 并验证形状
    ct_array = sitk.GetArrayFromImage(ct_image)
    pet_array = sitk.GetArrayFromImage(pet_registered)

    print(f"\nFinal shapes:")
    print(f"  CT:  {ct_array.shape}")
    print(f"  PET: {pet_array.shape}")

    # 验证形状匹配
    if ct_array.shape == pet_array.shape:
        print("  ✓ Shapes match!")
    else:
        print("  ✗ Shapes do not match!")

    # 6. 保存 NPY 和 NIfTI 文件
    ct_npy, pet_npy, ct_nii, pet_nii = save_as_npy_and_nii(
        ct_image, pet_registered, output_folder, patient_id
    )

    return ct_array.shape, pet_array.shape


def find_ct_folder(patient_folder):
    """
    在患者文件夹中查找 CT 文件夹 (包含 'RP_matched')
    """
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        if os.path.isdir(item_path) and 'RP_matched' in item:
            return item_path
    return None


def find_pet_folder(patient_folder):
    """
    在患者文件夹中查找 PET 主文件夹 (包含 '_PT_')
    返回 PET 主文件夹路径
    """
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        if os.path.isdir(item_path) and '_PT_' in item:
            return item_path
    return None


def find_pet_dicom_folder(pt_folder):
    """
    在 _PT_ 文件夹内查找实际包含 PET DICOM 图像文件的文件夹
    
    策略：
    1. 先检查所有子文件夹
    2. 最后检查 _PT_ 文件夹本身
    3. 返回第一个包含 PT 模态图像的文件夹
    """
    def count_pet_images(folder):
        """
        统计文件夹中 PET 图像文件的数量
        """
        count = 0
        try:
            files = os.listdir(folder)
            
            for f in files:
                if not f.lower().endswith('.dcm'):
                    continue
                
                filepath = os.path.join(folder, f)
                
                try:
                    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                    
                    # 只统计 PT (PET) 模态的文件
                    if hasattr(ds, 'Modality') and ds.Modality == 'PT':
                        count += 1
                    
                except:
                    continue
            
        except:
            pass
        
        return count
    
    # 收集所有候选文件夹（子文件夹 + 当前文件夹）
    candidate_folders = []
    
    # 先添加子文件夹
    try:
        for item in os.listdir(pt_folder):
            item_path = os.path.join(pt_folder, item)
            if os.path.isdir(item_path):
                candidate_folders.append(item_path)
    except:
        pass
    
    # 最后添加当前文件夹
    candidate_folders.append(pt_folder)
    
    # 查找第一个包含 PET 图像的文件夹
    for folder in candidate_folders:
        pet_count = count_pet_images(folder)
        
        if pet_count > 0:
            folder_name = os.path.basename(folder) if folder != pt_folder else "_PT_ folder itself"
            print(f"    Found {pet_count} PET images in: {folder_name}")
            return folder
    
    # 如果都没找到
    print(f"    Warning: No PET image files found in _PT_ folder or its subfolders")
    return None


def find_reg_file(pt_folder):
    """
    在 _PT_ 文件夹中查找 REG 文件
    """
    for item in os.listdir(pt_folder):
        item_path = os.path.join(pt_folder, item)

        # REG 文件可能是单个 .dcm 文件或文件夹
        if os.path.isfile(item_path) and 'RE' in item.upper():
            return item_path

        if os.path.isdir(item_path) and 'RE' in item.upper():
            # 查找文件夹中的 dcm 文件
            for f in os.listdir(item_path):
                if f.endswith('.dcm'):
                    return os.path.join(item_path, f)

    return None


def process_all_patients(
    base_folder,
    output_folder,
    method='resample_only',
    use_inverse_transform=True,
    try_both_directions=False,
    filter_by_npy=False
):
    """
    处理指定文件夹中的所有患者

    Args:
        base_folder: 包含所有患者文件夹的基础路径
        output_folder: 输出文件夹
        method: 处理方法 ('resample_only' 或 'apply_transform')
        use_inverse_transform: 如果REG文件是CT->PET,设为True获取PET->CT的逆变换
        try_both_directions: 如果True,会尝试正向和逆向两种变换并选择最好的
        filter_by_npy: 如果True, 只处理 NPY_FILTER_FOLDER 中存在对应 _PET.npy
                       文件的患者。Patient ID 通过文件名提取，例如
                       GYNDATASET011_PET.npy -> GYNDATASET011，然后与患者子文件夹
                       名称进行匹配。如果False, 处理所有患者（原有行为）。
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # ---------- 如果开启过滤，提前读取目标患者集合 ----------
    allowed_patient_ids = None   # None 意味着不过滤
    if filter_by_npy:
        print(f"\n[filter_by_npy = True]")
        print(f"  Scanning NPY folder: {NPY_FILTER_FOLDER}")
        allowed_patient_ids = get_patient_ids_from_npy_folder(NPY_FILTER_FOLDER)
        print(f"  Found {len(allowed_patient_ids)} target patients from .npy files:")
        for pid in sorted(allowed_patient_ids):
            print(f"    - {pid}")
        print()

    # 获取所有患者文件夹
    patient_folders = [f for f in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, f))]
    
    print(f"\nFound {len(patient_folders)} patient folders in base directory")
    print(f"{'='*60}\n")

    # 统计信息
    success_count = 0
    failed_patients = []
    skipped_patients = []

    # 处理每个患者
    for patient_id in patient_folders:
        patient_folder = os.path.join(base_folder, patient_id)

        # ---------- 过滤逻辑：如果开启，跳过不在目标集合中的患者 ----------
        if allowed_patient_ids is not None and patient_id not in allowed_patient_ids:
            skipped_patients.append(patient_id)
            continue
        
        try:
            # 查找 CT 文件夹
            ct_folder = find_ct_folder(patient_folder)
            if ct_folder is None:
                print(f"⚠ {patient_id}: No CT folder found (looking for 'RP_matched')")
                failed_patients.append((patient_id, "No CT folder"))
                continue

            # 查找 PET 主文件夹 (_PT_)
            pt_main_folder = find_pet_folder(patient_folder)
            if pt_main_folder is None:
                print(f"⚠ {patient_id}: No PET main folder found (looking for '_PT_')")
                failed_patients.append((patient_id, "No PET main folder"))
                continue

            print(f"✓ {patient_id}: Found _PT_ folder: {os.path.basename(pt_main_folder)}")

            # 在 _PT_ 文件夹内查找实际的 PET DICOM 文件夹
            pet_dicom_folder = find_pet_dicom_folder(pt_main_folder)
            if pet_dicom_folder is None:
                print(f"⚠ {patient_id}: No PET DICOM files found")
                failed_patients.append((patient_id, "No PET DICOM files"))
                continue

            # 在 _PT_ 文件夹内查找 REG 文件
            reg_file = find_reg_file(pt_main_folder)
            if reg_file:
                print(f"✓ {patient_id}: Found REG file")
            else:
                print(f"✓ {patient_id}: No REG file (will use resample only)")

            # 处理患者
            process_single_patient(
                ct_folder=ct_folder,
                pet_folder=pet_dicom_folder,
                reg_file=reg_file,
                output_folder=output_folder,
                patient_id=patient_id,
                method=method,
                use_inverse_transform=use_inverse_transform,
                try_both_directions=try_both_directions
            )
            
            success_count += 1
            print(f"✓ {patient_id}: Successfully processed\n")

        except Exception as e:
            print(f"✗ {patient_id}: Failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_patients.append((patient_id, str(e)))
            continue

    # 打印总结
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total patient folders in base dir : {len(patient_folders)}")
    if allowed_patient_ids is not None:
        print(f"Target patients (from .npy files) : {len(allowed_patient_ids)}")
        print(f"Skipped (not in .npy list)         : {len(skipped_patients)}")
    print(f"Successfully processed             : {success_count}")
    print(f"Failed                             : {len(failed_patients)}")
    
    if failed_patients:
        print(f"\nFailed patients:")
        for patient_id, reason in failed_patients:
            print(f"  - {patient_id}: {reason}")

    # 如果开启过滤，检查是否有 npy 中的 ID 在 base_folder 里找不到
    if allowed_patient_ids is not None:
        found_ids = set(patient_folders)
        missing_ids = allowed_patient_ids - found_ids
        if missing_ids:
            print(f"\n⚠ WARNING: The following patient IDs from .npy files have no")
            print(f"  matching subfolder in the base directory:")
            for mid in sorted(missing_ids):
                print(f"    - {mid}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # ========== 配置 - 修改这里 ==========

    # 包含所有患者文件夹的基础路径
    BASE_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data"

    # 输出文件夹
    OUTPUT_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\registered_all"

    # 处理方法: 'resample_only' 或 'apply_transform'
    METHOD = 'apply_transform'

    # 是否使用逆变换 (REG文件是CT->PET,需要PET->CT时设为True)
    USE_INVERSE_TRANSFORM = True

    # 是否自动尝试两个方向并选择最好的结果
    TRY_BOTH_DIRECTIONS = True  # 推荐先设为True进行诊断

    # 是否只处理 NPY_FILTER_FOLDER 中存在对应 _PET.npy 的患者
    # True  -> 只处理那些在 normalized_PET_invalid_mean 文件夹中有 *_PET.npy 的患者
    # False -> 处理 BASE_FOLDER 中的所有患者（原有行为）
    FILTER_BY_NPY = True

    # ========== 运行 ==========

    process_all_patients(
        base_folder=BASE_FOLDER,
        output_folder=OUTPUT_FOLDER,
        method=METHOD,
        use_inverse_transform=USE_INVERSE_TRANSFORM,
        try_both_directions=TRY_BOTH_DIRECTIONS,
        filter_by_npy=FILTER_BY_NPY
    )