**`addmask_fixct80MIL_paironlynomiss_dual_branch_mil_recurrence.log`**                                                 
  Paired-only baseline. Dual-branch MIL model (CT + PET) trained and validated exclusively on CT+PET paired patients (no
  CT-only subjects). No missing gate, no PET dropout. Target label: recurrence. 5-fold CV, base model size.              
                                                                                                                         
**`addmask_fixct80MIL_aux02drop0_missing_gate_mil_recurrence.log`**                                                    
  Missing-gate joint training. Dual-branch MIL model with a missing-aware PET gate trained on all patients (paired CT+PET+ CT-only), auxiliary loss weight = 0.2, PET dropout = 0. Validated on paired patients only; per-epoch AUC reported separately for paired vs. CT-only subgroups and for zeroed-PET inference. Target label: recurrence. 5-fold CV, base model size.    
