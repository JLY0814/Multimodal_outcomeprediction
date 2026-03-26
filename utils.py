"""
Utility functions: patch extraction, balanced sampler, metrics.
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from sklearn.metrics import roc_curve, roc_auc_score
from typing import List, Tuple, Optional
import random


# ── Patch extraction ──────────────────────────────────────────────────────────

def get_roi_center(mask: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute the centroid of a binary mask array (z, y, x order).
    Falls back to bounding-box center if mask is empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        # fallback: image center
        return tuple(s // 2 for s in mask.shape)
    centroid = coords.mean(axis=0)
    return tuple(int(round(c)) for c in centroid)


def extract_patch(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_mode: str = "reflect",
) -> np.ndarray:
    """
    Extract a 3-D patch of `patch_size` (D, H, W) centred on `center`
    from `volume` (D, H, W). Out-of-bounds regions are padded.

    Args:
        volume:     3-D numpy array (D, H, W).
        center:     (z, y, x) voxel index of the patch center.
        patch_size: (pd, ph, pw) voxel extents of the output patch.
        pad_mode:   numpy pad mode ('reflect', 'constant', …).

    Returns:
        patch:  numpy array of shape patch_size.
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size

    # half-sizes (round down for even dims so centre is top-left of middle)
    hd, hh, hw = pd // 2, ph // 2, pw // 2

    z0, z1 = center[0] - hd, center[0] - hd + pd
    y0, y1 = center[1] - hh, center[1] - hh + ph
    x0, x1 = center[2] - hw, center[2] - hw + pw

    # compute required padding
    pad_z0 = max(0, -z0);  pad_z1 = max(0, z1 - D)
    pad_y0 = max(0, -y0);  pad_y1 = max(0, y1 - H)
    pad_x0 = max(0, -x0);  pad_x1 = max(0, x1 - W)

    if any(p > 0 for p in [pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1]):
        volume = np.pad(
            volume,
            ((pad_z0, pad_z1), (pad_y0, pad_y1), (pad_x0, pad_x1)),
            mode=pad_mode,
        )
        # shift indices after padding
        z0 += pad_z0; z1 += pad_z0
        y0 += pad_y0; y1 += pad_y0
        x0 += pad_x0; x1 += pad_x0

    return volume[z0:z1, y0:y1, x0:x1]


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalise_ct(patch: np.ndarray,
                 wl: float = 40.0,
                 ww: float = 400.0) -> np.ndarray:
    """Window-level clip + [0,1] normalisation for CT (HU values)."""
    lo, hi = wl - ww / 2, wl + ww / 2
    patch = np.clip(patch, lo, hi)
    return (patch - lo) / (hi - lo)


def normalise_pet(patch: np.ndarray, max_suv: float = 10.0) -> np.ndarray:
    """SUV normalisation: clip to [0, max_suv], then scale to [0, 1]."""
    patch = np.clip(patch, 0.0, max_suv)
    return patch / max_suv


def normalise_dose(patch: np.ndarray, max_gy: float = 70.0) -> np.ndarray:
    """Dose normalisation: clip to [0, max_gy] Gy, then scale to [0, 1]."""
    patch = np.clip(patch, 0.0, max_gy)
    return patch / max_gy


# ── Balanced batch sampler ────────────────────────────────────────────────────

class BalancedBatchSampler(Sampler):
    """
    Yields batches with exactly batch_size // 2 positives and negatives.
    Samples with replacement from the minority class if needed.
    """

    def __init__(self, labels: List[int], batch_size: int):
        super().__init__(None)
        assert batch_size % 2 == 0, "batch_size must be even"
        self.labels     = np.array(labels)
        self.batch_size = batch_size
        self.pos_idx    = np.where(self.labels == 1)[0].tolist()
        self.neg_idx    = np.where(self.labels == 0)[0].tolist()
        n_batches       = max(len(self.pos_idx), len(self.neg_idx)) * 2 // batch_size
        self.n_batches  = max(n_batches, 1)

    def __iter__(self):
        half = self.batch_size // 2
        pos_pool = self.pos_idx.copy()
        neg_pool = self.neg_idx.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        def cycle(pool, n):
            """sample n items from pool, cycling if needed."""
            out = []
            while len(out) < n:
                if len(pool) == 0:
                    pool = self.pos_idx.copy() if pool is pos_pool else self.neg_idx.copy()
                    random.shuffle(pool)
                out.append(pool.pop())
            return out

        for _ in range(self.n_batches):
            batch = cycle(pos_pool, half) + cycle(neg_pool, half)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


# ── Metrics & threshold selection ─────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray,
                            y_prob: np.ndarray) -> float:
    """
    Find the probability threshold that maximises the Youden index
    (sensitivity + specificity - 1) on the supplied data.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: Optional[float] = None
                    ) -> dict:
    """
    Compute AUC, sensitivity, specificity at a given threshold.
    If threshold is None, it is derived via Youden index.
    """
    auc = roc_auc_score(y_true, y_prob)
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    sensitivity  = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    ppv          = tp / (tp + fp + 1e-8)
    npv          = tn / (tn + fn + 1e-8)
    accuracy     = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return dict(
        auc=auc,
        threshold=threshold,
        sensitivity=sensitivity,
        specificity=specificity,
        ppv=ppv,
        npv=npv,
        accuracy=accuracy,
        tp=tp, fn=fn, tn=tn, fp=fp,
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
