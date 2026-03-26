"""
Training loop for one cross-validation fold.

Usage (called by main.py; can also be run standalone for debugging):
    python train.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from typing import Optional

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    LR_PATIENCE, EARLY_STOP, MIN_EPOCHS, AMP,
    CT_FINETUNE_EPOCHS, CT_FINETUNE_LR, CT_AUX_WEIGHT,
    DLA_CHANNELS, EMBED_DIM, FUSION_HIDDEN, DROPOUT,
    MIXUP_PROB, MIXUP_ALPHA,
    USE_DOSE, DOSE_CHANNELS,
    USE_MIL, MIL_ATTN_DIM,
    OUTPUT_DIR,
    CT_LARGE_CHANNELS, DOSE_LARGE_CHANNELS, DUAL_SCALE_FUSION_HIDDEN,
    DUAL_SCALE_USE_CT_LARGE,
    MISSING_GATE_HIDDEN,
    GLOBAL_ENC_CHANNELS, GLOBAL_OUT_DIM, DOSE_ALPHA_INIT, FUSION_GATE_HIDDEN,
)
from model import DualBranch3DCNN, DualScaleModel, CombinedPETCTDoseMILModel
from dataset import PatchDataset
from sklearn.metrics import roc_auc_score
from utils import BalancedBatchSampler, find_optimal_threshold, compute_metrics


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying model, unwrapping DDP if necessary."""
    return model.module if isinstance(model, DDP) else model


def _bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                      n_boot: int = 1000, ci: float = 0.95,
                      seed: int = 0) -> tuple:
    """
    Return (lower, upper) bootstrap CI for AUC.
    Returns (nan, nan) if there are fewer than 2 samples or only one class.
    """
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    rng   = np.random.default_rng(seed)
    n     = len(y_true)
    aucs  = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) < 10:
        return float("nan"), float("nan")
    alpha = (1 - ci) / 2
    return float(np.percentile(aucs, alpha * 100)), float(np.percentile(aucs, (1 - alpha) * 100))


# ── MIL attention auxiliary loss ──────────────────────────────────────────────

def mil_auxiliary_loss(
    A:               torch.Tensor,   # (B, N, 1) softmax attention weights
    patch_types:     torch.Tensor,   # (B, N) LongTensor: 0=tumor / 1=background
    sparsity_weight: float = 0.0,
    rank_weight:     float = 0.0,
    rank_margin:     float = 0.1,
) -> torch.Tensor:
    """
    Attention regularisation losses:
      - Entropy minimisation (sparsity): encourages peaked attention.
      - Rank constraint: tumor patches should receive higher attention than bg.
    """
    loss = A.new_zeros(1).squeeze()
    A_sq = A.squeeze(-1)             # (B, N)

    if sparsity_weight > 0:
        entropy = -(A_sq * (A_sq + 1e-8).log()).sum(dim=1).mean()
        loss    = loss + sparsity_weight * entropy

    if rank_weight > 0 and patch_types is not None:
        tumor_m = (patch_types == 0).float()
        bg_m    = (patch_types == 1).float()
        n_t     = tumor_m.sum(1).clamp(min=1)
        n_b     = bg_m.sum(1).clamp(min=1)
        mean_t  = (A_sq * tumor_m).sum(1) / n_t
        mean_b  = (A_sq * bg_m).sum(1) / n_b
        # hinge: max(0, mean_bg - mean_tumor + margin)
        loss    = loss + rank_weight * torch.clamp(
            mean_b - mean_t + rank_margin, min=0).mean()

    return loss


# ── Training step ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool,
    use_dual_scale: bool = False,
    use_ct_only: bool = False,
    use_pet_only: bool = False,
    use_bg_patches: bool = False,
    attn_sparsity_weight: float = 0.0,
    rank_constraint_weight: float = 0.0,
    rank_margin: float = 0.1,
    use_missing_gate: bool = False,
    use_global_branch: bool = False,
    ct_aux_weight: float = 0.0,
    use_unipair: bool = False,
) -> float:
    model.train()
    total_loss = 0.0

    if use_dual_scale:
        for ct_small, pet_small, ct_large, dose_large, labels in tqdm(
                loader, desc="  train", leave=False):
            ct_small   = ct_small.to(device)
            pet_small  = pet_small.to(device)
            ct_large   = ct_large.to(device)
            dose_large = dose_large.to(device)
            labels     = labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(ct_small, pet_small, ct_large, dose_large)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * ct_small.size(0)
    elif use_ct_only:
        for ct, labels in tqdm(loader, desc="  train", leave=False):
            ct     = ct.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(ct)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * ct.size(0)
    elif use_pet_only:
        for pet, labels in tqdm(loader, desc="  train", leave=False):
            pet    = pet.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(pet)   # model receives PET as first positional arg
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * pet.size(0)
    elif use_unipair:
        # ── Unipair path: (ct, pet, dose, labels, modality) ──────────────────
        for batch in tqdm(loader, desc="  train", leave=False):
            ct         = batch[0].to(device)
            pet        = batch[1].to(device)
            dose       = batch[2].to(device)
            labels     = batch[3].to(device)
            modality_b = batch[4].to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model(ct, pet, dose, modality=modality_b)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * ct.size(0)
    else:
        need_attn = (use_bg_patches and
                     (attn_sparsity_weight > 0 or rank_constraint_weight > 0))
        for batch in tqdm(loader, desc="  train", leave=False):
            # Strip from the end in a fixed order:
            #   1. pet_present  (always last when use_missing_gate)
            #   2. ct_global, dose_global (when use_global_branch)
            # Remaining: (ct, pet, dose_roi, label [, patch_types])
            _b = list(batch)
            pet_present_b = _b.pop().to(device) if use_missing_gate else None
            if use_global_branch:
                dose_global_b = _b.pop().to(device)
                ct_global_b   = _b.pop().to(device)
            else:
                ct_global_b = dose_global_b = None

            if use_bg_patches:
                ct, pet, dose, labels, patch_types = _b
                patch_types = patch_types.to(device)
            else:
                ct, pet, dose, labels = _b
                patch_types = None

            ct     = ct.to(device)
            pet    = pet.to(device)
            dose   = dose.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                if need_attn:
                    # bg_patches MIL: need attention weights for aux MIL loss
                    if use_missing_gate and ct_aux_weight > 0:
                        if use_global_branch:
                            logits, A, logits_ct_aux = model(
                                ct, pet, ct_global_b, dose_global_b,
                                pet_present=pet_present_b,
                                return_attn=True, return_ct_logits=True)
                        else:
                            logits, A, logits_ct_aux = model(
                                ct, pet, dose,
                                return_attn=True, return_ct_logits=True,
                                pet_present=pet_present_b)
                    else:
                        if use_global_branch:
                            logits, A = model(ct, pet, ct_global_b, dose_global_b,
                                              pet_present=pet_present_b,
                                              return_attn=True)
                        else:
                            logits, A = model(ct, pet, dose,
                                              return_attn=True,
                                              pet_present=pet_present_b)
                    task_loss = criterion(logits, labels)
                    aux_mil   = mil_auxiliary_loss(A, patch_types,
                                                   attn_sparsity_weight,
                                                   rank_constraint_weight,
                                                   rank_margin)
                    if use_missing_gate and ct_aux_weight > 0:
                        aux_ct = criterion(logits_ct_aux, labels)
                        loss   = task_loss + aux_mil + ct_aux_weight * aux_ct
                    else:
                        loss = task_loss + aux_mil
                else:
                    if use_missing_gate and ct_aux_weight > 0:
                        if use_global_branch:
                            logits, logits_ct_aux = model(
                                ct, pet, ct_global_b, dose_global_b,
                                pet_present=pet_present_b,
                                return_ct_logits=True)
                        else:
                            logits, logits_ct_aux = model(
                                ct, pet, dose,
                                pet_present=pet_present_b,
                                return_ct_logits=True)
                        main_loss = criterion(logits, labels)
                        aux_ct    = criterion(logits_ct_aux, labels)
                        loss      = main_loss + ct_aux_weight * aux_ct
                    else:
                        if use_global_branch:
                            logits = model(ct, pet, ct_global_b, dose_global_b,
                                           pet_present=pet_present_b)
                        else:
                            logits = model(ct, pet, dose, pet_present=pet_present_b)
                        loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * ct.size(0)

    return total_loss / len(loader.dataset)


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_dual_scale: bool = False,
    use_ct_only: bool = False,
    use_pet_only: bool = False,
    use_missing_gate: bool = False,
    use_global_branch: bool = False,
    force_drop_pet: bool = False,
    use_unipair: bool = False,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Returns (y_true, y_prob, pet_present_arr).
    pet_present_arr is None unless use_missing_gate=True or use_unipair=True
    (in the unipair case it holds the modality values: 0=CT, 1=PET).
    force_drop_pet=True zeroes PET and sets pet_present=0 for all samples."""
    model.eval()
    all_probs       = []
    all_labels      = []
    all_pet_present = [] if (use_missing_gate or use_unipair) else None

    if use_unipair:
        for batch in tqdm(loader, desc="  val  ", leave=False):
            ct         = batch[0].to(device)
            pet        = batch[1].to(device)
            dose       = batch[2].to(device)
            labels     = batch[3]
            modality_b = batch[4].to(device)
            logits = model(ct, pet, dose, modality=modality_b)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_pet_present.extend(batch[4].numpy().tolist())

        pet_arr = np.array(all_pet_present)
        return np.array(all_labels), np.array(all_probs), pet_arr

    if use_dual_scale:
        for ct_small, pet_small, ct_large, dose_large, labels in tqdm(
                loader, desc="  val  ", leave=False):
            logits = model(
                ct_small.to(device), pet_small.to(device),
                ct_large.to(device), dose_large.to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    elif use_ct_only:
        for ct, labels in tqdm(loader, desc="  val  ", leave=False):
            logits = model(ct.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    elif use_pet_only:
        for pet, labels in tqdm(loader, desc="  val  ", leave=False):
            logits = model(pet.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    else:
        for batch in tqdm(loader, desc="  val  ", leave=False):
            _b = list(batch)
            pet_present_b = _b.pop().to(device) if use_missing_gate else None
            if use_missing_gate:
                all_pet_present.extend(batch[-1].numpy().tolist())
            if use_global_branch:
                dose_global_b = _b.pop().to(device)
                ct_global_b   = _b.pop().to(device)
            else:
                ct_global_b = dose_global_b = None

            ct     = _b[0].to(device)
            pet    = _b[1].to(device)
            if force_drop_pet:
                pet = torch.zeros_like(pet)
                if use_missing_gate:
                    pet_present_b = torch.zeros(pet.shape[0], device=device)
            dose   = _b[2].to(device)
            labels = _b[3]

            if use_global_branch:
                logits = model(ct, pet, ct_global_b, dose_global_b,
                               pet_present=pet_present_b)
            else:
                logits = model(ct, pet, dose, pet_present=pet_present_b)

            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    pet_arr = np.array(all_pet_present) if all_pet_present is not None else None
    return np.array(all_labels), np.array(all_probs), pet_arr


# ── Main training function ────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    train_dataset: PatchDataset,
    val_dataset: PatchDataset,
    device: torch.device,
    save_dir: str = OUTPUT_DIR,
    use_dose: bool = False,
    use_mil:  bool = False,
    dla_channels:  list = None,
    fusion_hidden: int  = None,
    dose_channels: list = None,
    use_dual_scale:           bool = False,
    ct_large_channels:        list = None,
    dose_large_channels:      list = None,
    dual_scale_fusion_hidden: int  = None,
    large_branch_use_ct:      bool = False,
    use_ct_only:              bool = False,
    use_pet_only:             bool = False,
    use_bg_patches:           bool  = False,
    attn_sparsity_weight:     float = 0.0,
    rank_constraint_weight:   float = 0.0,
    rank_margin:              float = 0.1,
    use_missing_gate:         bool  = False,
    use_global_branch:        bool  = False,
    use_ct_finetune:          bool  = False,
    ctonly_dataset:           PatchDataset = None,
    eval_drop_pet:            bool  = False,
    ct_aux_weight:            float = 0.0,
    use_unipair:              bool  = False,
    rank:        int = 0,
    world_size:  int = 1,
    num_workers: int = 4,
) -> dict:
    """
    Train for one cross-validation fold.

    In multi-GPU (DDP) mode:
      - rank 0  : runs validation, saves checkpoint, prints logs.
      - all ranks: share training data via DistributedSampler, gradients
                   are synchronised automatically by DDP.

    Returns a dict on rank 0; returns None on other ranks.
    """
    is_dist = world_size > 1
    is_main = (rank == 0)

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"fold{fold_idx}_best.pth")

    # ── Data loaders ──────────────────────────────────────────────────────────
    if is_dist:
        # DistributedSampler splits the dataset across GPUs.
        # BalancedBatchSampler is incompatible with DDP; pos_weight handles
        # class imbalance instead.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,   # avoid batch-size-1 which breaks BatchNorm1d in DDP
            persistent_workers=num_workers > 0,
        )
    else:
        train_sampler = None
        sampler = BalancedBatchSampler(train_dataset.label_list, BATCH_SIZE)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # Validation loader only needed on rank 0
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    # ── Model, optimiser, scheduler ───────────────────────────────────────────
    _dla_channels  = dla_channels  if dla_channels  is not None else DLA_CHANNELS
    _fusion_hidden = fusion_hidden if fusion_hidden is not None else FUSION_HIDDEN
    _dose_channels = dose_channels if dose_channels is not None else DOSE_CHANNELS

    if use_global_branch:
        model = CombinedPETCTDoseMILModel(
            dla_channels=_dla_channels,
            fusion_hidden=_fusion_hidden,
            dropout=DROPOUT,
            use_mil=use_mil,
            mil_attn_dim=MIL_ATTN_DIM,
            use_missing_gate=use_missing_gate,
            missing_gate_hidden=MISSING_GATE_HIDDEN,
            global_channels=GLOBAL_ENC_CHANNELS,
            global_out_dim=GLOBAL_OUT_DIM,
            dose_alpha_init=DOSE_ALPHA_INIT,
            gate_hidden=FUSION_GATE_HIDDEN,
        ).to(device)
    elif use_dual_scale:
        _ct_large_channels        = ct_large_channels        if ct_large_channels        is not None else CT_LARGE_CHANNELS
        _dose_large_channels      = dose_large_channels      if dose_large_channels      is not None else DOSE_LARGE_CHANNELS
        _dual_scale_fusion_hidden = dual_scale_fusion_hidden if dual_scale_fusion_hidden is not None else DUAL_SCALE_FUSION_HIDDEN
        model = DualScaleModel(
            dla_channels=_dla_channels,
            fusion_hidden=_fusion_hidden,
            dropout=DROPOUT,
            use_mil=use_mil,
            mil_attn_dim=MIL_ATTN_DIM,
            ct_large_channels=_ct_large_channels,
            dose_large_channels=_dose_large_channels,
            dual_scale_fusion_hidden=_dual_scale_fusion_hidden,
            large_branch_use_ct=large_branch_use_ct,
        ).to(device)
    else:
        model = DualBranch3DCNN(
            dla_channels=_dla_channels,
            fusion_hidden=_fusion_hidden,
            dropout=DROPOUT,
            use_dose=use_dose,
            dose_channels=_dose_channels,
            use_mil=use_mil,
            mil_attn_dim=MIL_ATTN_DIM,
            ct_only=use_ct_only,
            pet_only=use_pet_only,
            use_missing_gate=use_missing_gate,
            gate_hidden=MISSING_GATE_HIDDEN,
            use_unipair=use_unipair,
        ).to(device)

    # Wrap with DDP after moving to device
    if is_dist:
        # torch.where(mod, pet_head, ct_head) always includes both heads in the
        # computation graph even when only one branch is selected, so every
        # parameter receives a gradient (possibly 0) on every step.
        # find_unused_parameters=False avoids the expensive autograd-graph
        # traversal that was causing DDP to hang at epoch boundaries.
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Weighted BCE to handle residual class imbalance within mixed batches
    n_pos = sum(train_dataset.label_list)
    n_neg = len(train_dataset) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=LR_PATIENCE, factor=0.5
    )
    scaler = GradScaler("cuda", enabled=AMP)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_auc         = -1.0
    best_threshold   = 0.5
    best_metrics     = {}
    no_improve_count = 0

    for epoch in range(1, EPOCHS + 1):
        # Ensure each epoch sees a different data permutation in DDP mode
        if is_dist:
            train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp=AMP, use_dual_scale=use_dual_scale,
            use_ct_only=use_ct_only,
            use_pet_only=use_pet_only,
            use_bg_patches=use_bg_patches,
            attn_sparsity_weight=attn_sparsity_weight,
            rank_constraint_weight=rank_constraint_weight,
            rank_margin=rank_margin,
            use_missing_gate=use_missing_gate,
            use_global_branch=use_global_branch,
            ct_aux_weight=ct_aux_weight,
            use_unipair=use_unipair,
        )

        # Average train loss across all ranks for consistent logging
        if is_dist:
            loss_t = torch.tensor(train_loss, device=device)
            dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            train_loss = loss_t.item()

        # Validation on rank 0; broadcast val AUC to all ranks for scheduler
        val_auc = 0.0
        metrics = {}
        if is_main:
            y_true, y_prob, pet_arr = evaluate(
                _unwrap(model), val_loader, device,
                use_dual_scale=use_dual_scale, use_ct_only=use_ct_only,
                use_pet_only=use_pet_only,
                use_missing_gate=use_missing_gate,
                use_global_branch=use_global_branch,
                use_unipair=use_unipair)
            metrics = compute_metrics(y_true, y_prob)
            val_auc = metrics["auc"]
            # Per-group AUCs for missing-gate mode
            if use_missing_gate and pet_arr is not None:
                for key, mask in [("paired", pet_arr == 1),
                                   ("ctonly", pet_arr == 0)]:
                    mask = mask.astype(bool)
                    if mask.sum() >= 2 and len(np.unique(y_true[mask])) > 1:
                        metrics[f"auc_{key}"] = float(
                            roc_auc_score(y_true[mask], y_prob[mask]))
                    else:
                        metrics[f"auc_{key}"] = float("nan")
            # Per-modality AUCs for unipair mode
            if use_unipair and pet_arr is not None:
                for key, mask in [("ct",  pet_arr == 0),
                                   ("pet", pet_arr == 1)]:
                    mask = mask.astype(bool)
                    if mask.sum() >= 2 and len(np.unique(y_true[mask])) > 1:
                        metrics[f"auc_{key}"] = float(
                            roc_auc_score(y_true[mask], y_prob[mask]))
                    else:
                        metrics[f"auc_{key}"] = float("nan")

        if is_dist:
            auc_t = torch.tensor(val_auc, device=device)
            dist.broadcast(auc_t, src=0)
            val_auc = auc_t.item()

        # All ranks step the scheduler with the same AUC
        scheduler.step(val_auc)

        if is_main:
            # Bootstrap 95% CI for overall AUC
            ci_lo, ci_hi = _bootstrap_auc_ci(y_true, y_prob)

            def _fmt_ci(v, lo, hi):
                if np.isnan(v):
                    return "  nan"
                if np.isnan(lo):
                    return f"{v:.4f}"
                return f"{v:.4f} [{lo:.3f},{hi:.3f}]"

            auc_extra = ""
            if use_missing_gate:
                auc_paired = metrics.get("auc_paired", float("nan"))
                auc_ctonly = metrics.get("auc_ctonly", float("nan"))
                # Per-group CIs
                paired_mask = (pet_arr == 1).astype(bool) if pet_arr is not None else np.ones(len(y_true), bool)
                ctonly_mask = (pet_arr == 0).astype(bool) if pet_arr is not None else np.zeros(len(y_true), bool)
                p_lo, p_hi = _bootstrap_auc_ci(y_true[paired_mask], y_prob[paired_mask])
                c_lo, c_hi = _bootstrap_auc_ci(y_true[ctonly_mask], y_prob[ctonly_mask])

                auc_extra = (
                    f"\n          AUC_paired {_fmt_ci(auc_paired, p_lo, p_hi)}"
                    f"  AUC_CTonly {_fmt_ci(auc_ctonly, c_lo, c_hi)}"
                )
            if use_unipair:
                auc_ct  = metrics.get("auc_ct",  float("nan"))
                auc_pet = metrics.get("auc_pet", float("nan"))
                ct_mask  = (pet_arr == 0).astype(bool) if pet_arr is not None else np.ones(len(y_true), bool)
                pet_mask = (pet_arr == 1).astype(bool) if pet_arr is not None else np.zeros(len(y_true), bool)
                ct_lo,  ct_hi  = _bootstrap_auc_ci(y_true[ct_mask],  y_prob[ct_mask])
                pet_lo, pet_hi = _bootstrap_auc_ci(y_true[pet_mask], y_prob[pet_mask])
                auc_extra += (
                    f"\n          AUC_CT  {_fmt_ci(auc_ct,  ct_lo,  ct_hi)}"
                    f"  AUC_PET {_fmt_ci(auc_pet, pet_lo, pet_hi)}"
                )

            def _fmt_ci_overall(v, lo, hi):
                if np.isnan(lo):
                    return f"{v:.4f}"
                return f"{v:.4f} [{lo:.3f},{hi:.3f}]"

            print(
                f"Fold {fold_idx} | Epoch {epoch:3d}/{EPOCHS} "
                f"| loss {train_loss:.4f} "
                f"| AUC {_fmt_ci_overall(val_auc, ci_lo, ci_hi)} "
                f"| sens {metrics.get('sensitivity', 0):.3f} "
                f"| spec {metrics.get('specificity', 0):.3f} "
                f"| thr {metrics.get('threshold', 0):.3f}"
                + auc_extra
            )
            # Print validation predicted probabilities
            prob_str = " ".join(f"{p:.3f}" for p in y_prob)
            print(f"          y_prob: [{prob_str}]")

            # CT-only ablation on paired val set (PET zeroed, monitoring only)
            # Not applicable to unipair mode (separate CT/PET heads, different eval loop).
            if eval_drop_pet and not use_ct_only and not use_pet_only and not use_dual_scale and not use_unipair:
                yt_dp, yp_dp, _ = evaluate(
                    _unwrap(model), val_loader, device,
                    use_missing_gate=use_missing_gate,
                    use_global_branch=use_global_branch,
                    force_drop_pet=True,
                )
                if len(yt_dp) >= 2 and len(np.unique(yt_dp)) > 1:
                    auc_dp = roc_auc_score(yt_dp, yp_dp)
                    dp_lo, dp_hi = _bootstrap_auc_ci(yt_dp, yp_dp)
                    print(f"          AUC_drop_pet {_fmt_ci_overall(auc_dp, dp_lo, dp_hi)}"
                          f"  (PET zeroed, paired val, monitor only)")

            if val_auc > best_auc:
                best_auc       = val_auc
                best_threshold = metrics["threshold"]
                best_metrics   = metrics
                torch.save(_unwrap(model).state_dict(), model_path)
                no_improve_count = 0
            else:
                no_improve_count += 1

        # Sync all ranks before broadcasting early-stop decision.
        # Rank 0 may have spent time on bootstrap CI / checkpoint save while
        # other ranks were idle; the barrier prevents NCCL state from drifting.
        if is_dist:
            dist.barrier()

        # Broadcast early-stop decision from rank 0 to all ranks
        # Early stopping cannot trigger before MIN_EPOCHS.
        if is_dist:
            stop_t = torch.tensor(
                int(epoch >= MIN_EPOCHS and no_improve_count >= EARLY_STOP) if is_main else 0,
                device=device)
            dist.broadcast(stop_t, src=0)
            should_stop = bool(stop_t.item())
        else:
            should_stop = epoch >= MIN_EPOCHS and no_improve_count >= EARLY_STOP

        if should_stop:
            if is_main:
                print(f"  Early stopping at epoch {epoch}.")
            break

    if is_main:
        print(f"\nFold {fold_idx} best val AUC: {best_auc:.4f}  "
              f"threshold: {best_threshold:.3f}")

    # ── CT-branch fine-tune phase ──────────────────────────────────────────────
    if use_ct_finetune and ctonly_dataset is not None and len(ctonly_dataset) > 0:
        if is_main:
            print(f"\n--- Fold {fold_idx} CT fine-tune phase "
                  f"({CT_FINETUNE_EPOCHS} epochs, {len(ctonly_dataset)} CT-only patients) ---")

        # Load best checkpoint from phase 1 into model
        _unwrap(model).load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True))

        # Only CT branch is trainable; everything else (PET branch, gate, heads) frozen.
        # head_ct is not used when test data is all paired, so no need to train it.
        for name, param in _unwrap(model).named_parameters():
            param.requires_grad = ('ct_branch' in name)

        # DDP requires all parameters to receive gradients by default.
        # Re-wrap with find_unused_parameters=True so frozen params are tolerated.
        if is_dist:
            raw_model = _unwrap(model)
            model = DDP(raw_model, device_ids=[rank], find_unused_parameters=True)

        ft_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, _unwrap(model).parameters()),
            lr=CT_FINETUNE_LR,
        )

        if is_dist:
            ft_sampler = DistributedSampler(
                ctonly_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            ft_loader = DataLoader(
                ctonly_dataset, batch_size=BATCH_SIZE, sampler=ft_sampler,
                num_workers=num_workers, pin_memory=True, drop_last=True,
                persistent_workers=num_workers > 0,
            )
        else:
            ft_loader = DataLoader(
                ctonly_dataset, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                persistent_workers=num_workers > 0,
            )

        ft_best_auc       = best_auc
        ft_best_threshold = best_threshold
        ft_best_metrics   = best_metrics
        ft_model_path     = os.path.join(
            os.path.dirname(model_path),
            f"fold{fold_idx}_best_ft.pth")

        for ft_epoch in range(1, CT_FINETUNE_EPOCHS + 1):
            if is_dist:
                ft_sampler.set_epoch(ft_epoch)
            model.train()
            ft_loss_sum, ft_n = 0.0, 0
            for batch in tqdm(ft_loader, desc=f"  ft-train e{ft_epoch}", leave=False):
                # ctonly_ds always returns 5-tuples (use_missing_gate=True in dataset)
                _b = list(batch)
                pet_present_b = _b.pop().to(device)   # always present
                ct     = _b[0].to(device)
                pet    = _b[1].to(device)              # zeros for CT-only patients
                dose   = _b[2].to(device)
                labels = _b[3].to(device)
                ft_optimizer.zero_grad()
                with autocast("cuda", enabled=AMP):
                    # pass pet_present only if model uses missing gate
                    logits = model(ct, pet, dose,
                                   pet_present=pet_present_b if use_missing_gate else None)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(ft_optimizer)
                scaler.update()
                ft_loss_sum += loss.item() * labels.size(0)
                ft_n += labels.size(0)

            ft_loss = ft_loss_sum / max(ft_n, 1)
            if is_dist:
                lt = torch.tensor(ft_loss, device=device)
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
                ft_loss = lt.item()

            if is_main:
                y_true_ft, y_prob_ft, pet_arr_ft = evaluate(
                    _unwrap(model), val_loader, device,
                    use_dual_scale=use_dual_scale, use_ct_only=use_ct_only,
                    use_pet_only=use_pet_only,
                    use_missing_gate=use_missing_gate,
                    use_global_branch=use_global_branch,
                    use_unipair=use_unipair)
                ft_metrics = compute_metrics(y_true_ft, y_prob_ft)
                ft_auc = ft_metrics["auc"]
                ci_lo, ci_hi = _bootstrap_auc_ci(y_true_ft, y_prob_ft)
                print(f"  [FT] Fold {fold_idx} | Epoch {ft_epoch:2d}/{CT_FINETUNE_EPOCHS} "
                      f"| loss {ft_loss:.4f} "
                      f"| AUC {ft_auc:.4f} [{ci_lo:.3f},{ci_hi:.3f}]")
                if ft_auc > ft_best_auc:
                    ft_best_auc       = ft_auc
                    ft_best_threshold = ft_metrics["threshold"]
                    ft_best_metrics   = ft_metrics
                    torch.save(_unwrap(model).state_dict(), ft_model_path)

            if is_dist:
                ft_auc_t = torch.tensor(
                    ft_best_auc if is_main else 0.0, device=device)
                dist.broadcast(ft_auc_t, src=0)

        # Unfreeze all params and restore original DDP wrapping (find_unused=False)
        for param in _unwrap(model).parameters():
            param.requires_grad = True
        if is_dist:
            raw_model = _unwrap(model)
            model = DDP(raw_model, device_ids=[rank], find_unused_parameters=False)

        if is_main:
            print(f"\nFold {fold_idx} fine-tune best AUC: {ft_best_auc:.4f}  "
                  f"threshold: {ft_best_threshold:.3f}")

    if is_main:
        return dict(
            fold=fold_idx,
            best_auc=best_auc,
            best_threshold=best_threshold,
            val_metrics=best_metrics,
            model_path=model_path,
            ft_best_auc=ft_best_auc if use_ct_finetune and ctonly_dataset is not None else None,
            ft_model_path=ft_model_path if use_ct_finetune and ctonly_dataset is not None else None,
        )
    return None  # non-main ranks don't need the result
