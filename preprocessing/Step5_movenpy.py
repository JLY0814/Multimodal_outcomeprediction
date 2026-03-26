import os
import shutil
from pathlib import Path

# Source directory
source_dir = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\masks_MTV_PTV"

# Destination directory (will be created in the same parent directory as source)
dest_dir = r"Z:\Active\DICOM_Databases\PlansByRP_CERVIX_EXPORT\all_patients_complete_data\masks_MTV_PTV\masks_added"

def move_npy_files(source, destination):
    """
    Move all .npy files from source directory to destination directory.
    Creates the destination directory if it doesn't exist.
    """
    # Create destination directory if it doesn't exist
    os.makedirs(destination, exist_ok=True)
    
    # Check if source directory exists
    if not os.path.exists(source):
        print(f"Error: Source directory does not exist: {source}")
        return
    
    # Count files moved
    files_moved = 0
    
    # Find and move all .npy files
    print(f"Searching for .npy files in: {source}")
    print(f"Moving to: {destination}\n")
    
    for filename in os.listdir(source):
        if filename.endswith('.npy'):
            source_file = os.path.join(source, filename)
            dest_file = os.path.join(destination, filename)
            
            try:
                shutil.move(source_file, dest_file)
                print(f"Moved: {filename}")
                files_moved += 1
            except Exception as e:
                print(f"Error moving {filename}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Total files moved: {files_moved}")
    print(f"{'='*50}")

if __name__ == "__main__":
    move_npy_files(source_dir, dest_dir)