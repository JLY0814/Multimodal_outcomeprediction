For preprocessing:
  Step1_regis_resam_CT_PET.py       — Register PET to CT, resample to CT spacing (DCM → NII)
  Step2_add_downsample_CT.py        — Downsample CT+PET NII to uniform spacing 0.9766×3mm (NII → NII+NPY)
  Step3_convert_contour2mask.py     — Convert RT structure contours to binary mask NII (DCM → NII)
  Step4_SUV_PET.py                  — Normalize PET NPY to SUV using mask + Excel patient data (NPY → NPY)
  Step5_movenpy.py                  — Move mask NPY files to organized dataset folder
  Step6_move_dosemap_npy.py         — Move dose map NPY files to organized dataset folder
  

  Non-step helper files for the dose map sub-pipeline:
  - harmonize_dose_map.py — normalize RD DICOM to prescription dose
  - resample_dosemap_2_CT.py — resample dose to CT grid
  - transfer_dm_to_npy.py — convert resampled dose DCM → NPY
  - nii_SUV_PET.py / convert_nii_2_npy.py 
