

  ## Code Overview                                                                                                       
                                                                                                                         
  ### Core Pipeline                                                                                                      
  | File | Description |                                                                                                 
  |------|-------------|                                                                                                 
  | `config.py` | All hyperparameter constants (patch sizes, model channels, training settings) |                        
  | `dataset.py` | `PatchDataset` — loads CT/PET/Dose volumes, extracts MIL patches, handles missing PET |               
  | `model.py` | Model definitions: `DualBranch3DCNN` (DLA3D + Gated Attention MIL), `DualScaleModel` |                  
  | `train.py` | Training loop, evaluation, MIL auxiliary loss |                                                         
  | `main.py` | Entry point — argparse, k-fold cross-validation pipeline |                                               
  | `evaluate.py` | Inference on trained checkpoints, ROC curve plotting, summary statistics |                           
  | `augmentation.py` | 3D augmentations: GridMask3D, MixUp3D, random flip/rotate |                                      
  | `utils.py` | Patch extraction, balanced sampler, AUC/metric utilities |                                              
  | `run.sh` | Bash launcher with flag switches for all training configurations |                                        
  | `run_4experiments.sh` | Parallel launcher for 4 missing-modality ablation experiments |                              
                                                                                                                         
  ### Analysis & Diagnostics                                                                                             
  | File | Description |                                                                                                 
  |------|-------------|                                                                                                 
  | `check_ct_distribution.py` | Compare CT embedding distributions between paired and CT-only patients (label stats +
  PCA/t-SNE) |                                                                                                           
  | `check_distribution_causes.py` | Investigate root causes of distribution shift: HU intensity, volume shape, metadata,
   tumour size |                                                                                                         
  | `check_figo_distribution.py` | Compare FIGO stage distribution between paired and CT-only patients |
  | `check_mask_axis.py` | Scan mask `.npy` files and flag potential axis order issues |                                 
  | `compare_logs.py` | Compare two `train.log` files fold-by-fold, outputs CSV and PNG summary |                        
  | `count_masks.py` | Count patients with valid MTV Cervix masks |                                                      
  | `mil_analysis.py` | MIL attention analysis: attention export, ring analysis, top-k patch removal, modality ablation |
  | `patch_size_comparison.py` | Compare AUC, prediction correlation, and attention variance across two patch-size       
  configs |                                                                                                              
                                                                                                                         
  ### Data Utilities                                                                                                     
  | File | Description |
  |------|-------------|
  | `fix_ct_axis.py` | Detect and fix CT `.npy` files stored in wrong axis order `(H,W,Z)` → `(Z,H,W)` |
  | `refix_ct_axis.py` | Re-transpose CT files from backup using corrected `(2,1,0)` permutation |                       
  | `visualize_npy.py` | Display mid-slices of two CT volumes along all three axes for orientation verification |        
  | `visualize_patch_mask.py` | Show CT patch ROI with mask overlay (axial/coronal/sagittal) for paired and CT-only      
  patients |                                                                                                             
  | `viz_patches.py` | Visualize tumor-center patches with mask overlay for the first N patients |

## Preprocessing Pipeline
  | File | Description |
  |------|-------------|                                                                                                 
  | `Step1_regis_resam_CT_PET.py` | Register PET to CT and resample to CT spacing (DCM → NII) |
  | `Step2_add_downsample_CT.py` | Downsample CT + PET to uniform spacing 0.9766 × 3 mm (NII → NII + NPY) |              
  | `Step3_convert_contour2mask.py` | Convert RT structure contours to binary mask (DCM → NII) |                         
  | `Step4_SUV_PET.py` | Normalize PET to SUV using tumor mask and patient metadata (NPY → NPY) |                        
  | `Step5_movenpy.py` | Move mask NPY files to organized dataset folder |                                               
  | `Step6_move_dosemap_npy.py` | Move dose map NPY files to organized dataset folder |                                  
                                                                                                                         
  ### Dose Map Sub-pipeline                                                                                              
  | File | Description |
  |------|-------------|                                                                                                 
  | `harmonize_dose_map.py` | Normalize RD DICOM to prescription dose |
  | `resample_dosemap_2_CT.py` | Resample dose map to CT grid |                                                          
  | `transfer_dm_to_npy.py` | Convert resampled dose DICOM to NPY |
  | `nii_SUV_PET.py` | SUV conversion helper operating on NII files |                                                    
  | `convert_nii_2_npy.py` | Convert NII volumes to NPY format |  
