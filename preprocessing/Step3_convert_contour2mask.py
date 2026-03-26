import os
import numpy as np
import SimpleITK as sitk
import pydicom
from pathlib import Path


def load_dicom_series(dicom_folder):
    """
    加载 DICOM 系列 (用于加载 CT 作为参考图像)

    Returns:
        image: SimpleITK Image
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_folder)

    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")

    reader.SetFileNames(dicom_files)
    image = reader.Execute()

    return image


def find_rs_file(patient_folder):
    """
    在患者文件夹中查找 RTSTRUCT 文件
    RS 文件通常在与 PET 文件夹同一层级
    """
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        
        # 检查是否是文件
        if os.path.isfile(item_path) and item_path.endswith('.dcm'):
            try:
                ds = pydicom.dcmread(item_path)
                if ds.Modality == 'RTSTRUCT':
                    return item_path
            except:
                continue
        
        # 检查是否是包含 RS 的文件夹
        if os.path.isdir(item_path) and ('RS' in item.upper() or 'STRUCT' in item.upper()):
            for f in os.listdir(item_path):
                if f.endswith('.dcm'):
                    file_path = os.path.join(item_path, f)
                    try:
                        ds = pydicom.dcmread(file_path)
                        if ds.Modality == 'RTSTRUCT':
                            return file_path
                    except:
                        continue
    
    return None


def extract_contours_from_rtstruct(rtstruct_file, reference_image):
    """
    从 RTSTRUCT 文件中提取轮廓并转换为 mask
    只提取包含 'MTV' 或 'PTV' 的轮廓

    Args:
        rtstruct_file: RTSTRUCT DICOM 文件路径
        reference_image: SimpleITK 参考图像 (CT)

    Returns:
        masks_dict: {contour_name: mask_array} 字典
    """
    # 读取 RTSTRUCT 文件
    ds = pydicom.dcmread(rtstruct_file)
    
    if ds.Modality != 'RTSTRUCT':
        raise ValueError(f"File is not RTSTRUCT, got {ds.Modality}")
    
    print(f"Found RTSTRUCT with {len(ds.StructureSetROISequence)} ROIs")
    
    # 获取参考图像信息
    size = reference_image.GetSize()
    spacing = reference_image.GetSpacing()
    origin = reference_image.GetOrigin()
    direction = reference_image.GetDirection()
    
    # 创建空的 mask 字典
    masks_dict = {}
    
    # 遍历所有 ROI
    for roi_seq in ds.StructureSetROISequence:
        roi_number = roi_seq.ROINumber
        roi_name = roi_seq.ROIName
        
        # 只处理包含 MTV 或 PTV 的轮廓
        if 'MTV' not in roi_name.upper() and 'PTV' not in roi_name.upper():
            print(f"  Skipping ROI: {roi_name} (not MTV/PTV)")
            continue
        
        print(f"  Processing ROI: {roi_name}")
        
        # 创建空的 mask
        mask_array = np.zeros(size[::-1], dtype=np.uint8)  # SimpleITK size 是 (x,y,z), numpy 是 (z,y,x)
        
        # 查找对应的 contour 数据
        contour_found = False
        for contour_seq in ds.ROIContourSequence:
            if contour_seq.ReferencedROINumber == roi_number:
                contour_found = True
                
                if not hasattr(contour_seq, 'ContourSequence'):
                    print(f"    Warning: No contour data for {roi_name}")
                    continue
                
                # 遍历每个切片的轮廓
                for contour in contour_seq.ContourSequence:
                    # 获取轮廓点
                    contour_data = contour.ContourData
                    num_points = len(contour_data) // 3
                    
                    # 转换为 (x, y, z) 坐标数组
                    points = np.array(contour_data).reshape(num_points, 3)
                    
                    # 将物理坐标转换为体素索引
                    # 使用 SimpleITK 的 TransformPhysicalPointToIndex
                    indices = []
                    for point in points:
                        try:
                            idx = reference_image.TransformPhysicalPointToIndex(point.tolist())
                            indices.append(idx)
                        except:
                            # 点在图像外
                            continue
                    
                    if len(indices) < 3:
                        continue
                    
                    indices = np.array(indices)
                    
                    # 确定 z 切片
                    z_slice = indices[0, 2]
                    
                    # 使用多边形填充
                    from skimage.draw import polygon
                    
                    # 提取 x, y 坐标
                    rr, cc = polygon(indices[:, 1], indices[:, 0], shape=size[1::-1])
                    
                    # 填充 mask
                    mask_array[z_slice, rr, cc] = 1
                
                break
        
        if not contour_found:
            print(f"    Warning: No contour sequence found for {roi_name}")
            continue
        
        # 保存到字典
        masks_dict[roi_name] = mask_array
        print(f"    Created mask with shape {mask_array.shape}, sum={mask_array.sum()}")
    
    return masks_dict


def save_masks_as_npy_and_nii(masks_dict, reference_image, output_folder, patient_id):
    """
    保存 masks 为 NPY 和 NIfTI 格式

    Args:
        masks_dict: {contour_name: mask_array} 字典
        reference_image: SimpleITK 参考图像 (用于获取空间信息)
        output_folder: 输出文件夹
        patient_id: 患者 ID

    Returns:
        saved_files: 保存的文件路径列表
    """
    saved_files = []
    
    for roi_name, mask_array in masks_dict.items():
        # 清理 ROI 名称以用作文件名
        safe_name = roi_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # ========== 保存 NPY 文件 ==========
        npy_path = os.path.join(output_folder, f"{patient_id}_{safe_name}.npy")
        np.save(npy_path, mask_array.astype(np.uint8))
        saved_files.append(npy_path)
        print(f"  Saved NPY: {npy_path}")
        
        # ========== 保存 NIfTI 文件 ==========
        # 将 numpy 数组转换为 SimpleITK Image
        mask_image = sitk.GetImageFromArray(mask_array)
        
        # 复制参考图像的空间信息
        mask_image.SetSpacing(reference_image.GetSpacing())
        mask_image.SetOrigin(reference_image.GetOrigin())
        mask_image.SetDirection(reference_image.GetDirection())
        
        # 保存为 NIfTI
        nii_path = os.path.join(output_folder, f"{patient_id}_{safe_name}.nii.gz")
        sitk.WriteImage(mask_image, nii_path)
        saved_files.append(nii_path)
        print(f"  Saved NIfTI: {nii_path}")
    
    return saved_files


def find_ct_folder(patient_folder):
    """
    在患者文件夹中查找 CT 文件夹 (包含 'RP_matched')
    """
    for item in os.listdir(patient_folder):
        item_path = os.path.join(patient_folder, item)
        if os.path.isdir(item_path) and 'RP_matched' in item:
            return item_path
    return None


def process_single_patient_masks(patient_folder, output_folder, patient_id):
    """
    处理单个患者的 RS 轮廓，转换为 masks

    Args:
        patient_folder: 患者文件夹路径
        output_folder: 输出文件夹
        patient_id: 患者 ID
    """
    print(f"\n{'='*60}")
    print(f"Processing: {patient_id}")
    print(f"{'='*60}")
    
    # 1. 查找 CT 文件夹 (作为参考图像)
    ct_folder = find_ct_folder(patient_folder)
    if ct_folder is None:
        raise ValueError(f"No CT folder found for {patient_id}")
    
    print(f"Found CT folder: {os.path.basename(ct_folder)}")
    
    # 2. 加载 CT 作为参考图像
    print("Loading CT as reference...")
    reference_image = load_dicom_series(ct_folder)
    print(f"  CT size: {reference_image.GetSize()}")
    print(f"  CT spacing: {reference_image.GetSpacing()}")
    print(f"  CT origin: {reference_image.GetOrigin()}")
    
    # 3. 查找 RTSTRUCT 文件
    rs_file = find_rs_file(patient_folder)
    if rs_file is None:
        raise ValueError(f"No RTSTRUCT file found for {patient_id}")
    
    print(f"Found RTSTRUCT file: {os.path.basename(rs_file)}")
    
    # 4. 提取轮廓并转换为 masks
    print("Extracting contours (MTV/PTV only)...")
    masks_dict = extract_contours_from_rtstruct(rs_file, reference_image)
    
    if len(masks_dict) == 0:
        print(f"  Warning: No MTV/PTV contours found for {patient_id}")
        return
    
    print(f"  Found {len(masks_dict)} MTV/PTV contours")
    
    # 5. 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 6. 保存 masks
    print("Saving masks...")
    saved_files = save_masks_as_npy_and_nii(
        masks_dict, reference_image, output_folder, patient_id
    )
    
    print(f"  Total files saved: {len(saved_files)}")


def process_all_patients_masks(base_folder, output_folder_name="masks"):
    """
    处理所有患者的 RS 轮廓

    Args:
        base_folder: 包含所有患者文件夹的基础路径
        output_folder_name: 输出文件夹名称 (将创建在 base_folder 下)
    """
    # 创建输出文件夹
    output_folder = os.path.join(base_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有患者文件夹
    patient_folders = [f for f in os.listdir(base_folder) 
                      if os.path.isdir(os.path.join(base_folder, f)) 
                      and f != output_folder_name]
    
    print(f"\nFound {len(patient_folders)} patient folders")
    print(f"Output folder: {output_folder}")
    print(f"{'='*60}\n")
    
    # 统计信息
    success_count = 0
    failed_patients = []
    
    # 处理每个患者
    for patient_id in patient_folders:
        patient_folder = os.path.join(base_folder, patient_id)
        
        try:
            process_single_patient_masks(patient_folder, output_folder, patient_id)
            success_count += 1
            print(f"✓ {patient_id}: Successfully processed")
            
        except Exception as e:
            print(f"✗ {patient_id}: Failed with error: {str(e)}")
            failed_patients.append((patient_id, str(e)))
            continue
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients: {len(patient_folders)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_patients)}")
    
    if failed_patients:
        print(f"\nFailed patients:")
        for patient_id, reason in failed_patients:
            print(f"  - {patient_id}: {reason}")


# ==================== 主程序 ====================

if __name__ == "__main__":
    # ========== 配置 - 修改这里 ==========
    
    # 包含所有患者文件夹的基础路径
    BASE_FOLDER = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data"
    
    # 输出文件夹名称 (将在 BASE_FOLDER 下创建)
    OUTPUT_FOLDER_NAME = "masks_MTV_PTV"
    
    # ========== 运行 ==========
    
    # 需要先安装: pip install scikit-image
    try:
        import skimage
    except ImportError:
        print("Error: scikit-image not installed")
        print("Please install it using: pip install scikit-image")
        exit(1)
    
    process_all_patients_masks(
        base_folder=BASE_FOLDER,
        output_folder_name=OUTPUT_FOLDER_NAME
    )