import numpy as np
import nibabel as nib
import os
from pathlib import Path

def convert_nii_to_npy(input_folder: str, output_folder: str = None):
    """
    Convert all .nii and .nii.gz files in input_folder to .npy format
    
    Args:
        input_folder: Path to folder containing NIfTI files
        output_folder: Path to output folder (default: creates 'npy_output' in input_folder)
    """
    
    print("\n==============================")
    print("📌 NIfTI to NPY CONVERTER")
    print("==============================")
    
    # -------- 1. Validate input folder --------
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # -------- 2. Create output folder --------
    if output_folder is None:
        output_path = input_path / "npy_output"
    else:
        output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n✅ Input folder: {input_path}")
    print(f"✅ Output folder: {output_path}")
    
    # -------- 3. Find all NIfTI files --------
    nii_files = list(input_path.glob("*.nii")) + list(input_path.glob("*.nii.gz"))
    
    if len(nii_files) == 0:
        print("\n❌ No .nii or .nii.gz files found in the input folder!")
        return
    
    print(f"\n📂 Found {len(nii_files)} NIfTI file(s)")
    print("------------------------------")
    
    # -------- 4. Convert each file --------
    success_count = 0
    error_count = 0
    
    for nii_file in nii_files:
        try:
            print(f"\n🔄 Processing: {nii_file.name}")
            
            # Load NIfTI file
            nii_img = nib.load(str(nii_file))
            data = nii_img.get_fdata()
            
            # Get file info
            original_shape = data.shape
            original_dtype = data.dtype
            
            print(f"   Shape: {original_shape}")
            print(f"   Original dtype: {original_dtype}")
            print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")
            
            # Create output filename (replace .nii or .nii.gz with .npy)
            if nii_file.suffix == '.gz':
                output_name = nii_file.stem.replace('.nii', '') + '.npy'
            else:
                output_name = nii_file.stem + '.npy'
            
            output_file = output_path / output_name
            
            # Save as NPY
            np.save(str(output_file), data)
            
            # Verify the saved file
            loaded_data = np.load(str(output_file))
            
            if loaded_data.shape == original_shape:
                print(f"   ✅ Saved: {output_name}")
                print(f"   ✅ Verified shape: {loaded_data.shape}")
                success_count += 1
            else:
                print(f"   ❌ Shape mismatch! Expected {original_shape}, got {loaded_data.shape}")
                error_count += 1
            
            # Calculate file sizes
            nii_size_mb = nii_file.stat().st_size / (1024 * 1024)
            npy_size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"   📊 Size: {nii_size_mb:.2f} MB (NIfTI) → {npy_size_mb:.2f} MB (NPY)")
            
        except Exception as e:
            print(f"   ❌ Error processing {nii_file.name}: {str(e)}")
            error_count += 1
    
    # -------- 5. Summary --------
    print("\n==============================")
    print("📌 CONVERSION SUMMARY")
    print("==============================")
    print(f"✅ Successfully converted: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Output location: {output_path}")
    print("==============================\n")


# ===============================
# ✅ Run Here
# ===============================
if __name__ == "__main__":
    # Input folder containing NIfTI files
    input_folder = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\masks_MTV_PTV_resampled_0.9766_3.0mm"
    
    # Output will be created in: input_folder/npy_output
    # Or you can specify a custom output folder:
    # output_folder = r"Z:\path\to\custom\output"
    
    convert_nii_to_npy(input_folder)
    
    # If you want custom output location, use:
    # convert_nii_to_npy(input_folder, output_folder)