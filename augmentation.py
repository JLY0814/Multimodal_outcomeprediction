"""
3-D augmentations: GridMask3D, MixUp3D, random flip/rotate.
All operations work on numpy arrays of shape (C, D, H, W).
"""

import numpy as np
import torch
import random
from typing import Tuple, Optional


# ── GridMask 3D ───────────────────────────────────────────────────────────────

class GridMask3D:
    """
    Randomly zero-out a grid of cubic blocks in a 3-D patch.

    Each block side length is drawn uniformly in [min_hole, max_hole].
    Blocks are placed on a regular grid with random offset and random
    gap between blocks. The operation is applied identically to every
    input channel, so CT and PET branches share the same spatial mask.

    Args:
        min_hole:  minimum block side length in voxels.
        max_hole:  maximum block side length in voxels.
        ratio:     expected fraction of masked voxels.
        prob:      probability of applying the transform.
    """

    def __init__(
        self,
        min_hole: int = 4,
        max_hole: int = 12,
        ratio: float = 0.4,
        prob: float = 0.5,
    ):
        self.min_hole = min_hole
        self.max_hole = max_hole
        self.ratio    = ratio
        self.prob     = prob

    def __call__(
        self,
        ct: np.ndarray,
        pet: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ct, pet: (C, D, H, W) numpy arrays.
        Returns augmented copies.
        """
        if random.random() > self.prob:
            return ct, pet

        _, D, H, W = ct.shape
        mask = self._make_mask(D, H, W)      # (D, H, W) boolean
        ct  = ct.copy();  ct[:, mask]  = 0.0
        pet = pet.copy(); pet[:, mask] = 0.0
        return ct, pet

    def _make_mask(self, D: int, H: int, W: int) -> np.ndarray:
        hole_d = random.randint(self.min_hole, self.max_hole)
        hole_h = random.randint(self.min_hole, self.max_hole)
        hole_w = random.randint(self.min_hole, self.max_hole)

        # gap chosen so that ratio ≈ ratio
        gap_d = max(1, int(hole_d / self.ratio) - hole_d)
        gap_h = max(1, int(hole_h / self.ratio) - hole_h)
        gap_w = max(1, int(hole_w / self.ratio) - hole_w)

        stride_d = hole_d + gap_d
        stride_h = hole_h + gap_h
        stride_w = hole_w + gap_w

        off_d = random.randint(0, stride_d - 1)
        off_h = random.randint(0, stride_h - 1)
        off_w = random.randint(0, stride_w - 1)

        mask = np.zeros((D, H, W), dtype=bool)
        for zs in range(off_d, D + stride_d, stride_d):
            for ys in range(off_h, H + stride_h, stride_h):
                for xs in range(off_w, W + stride_w, stride_w):
                    z0, z1 = zs, min(zs + hole_d, D)
                    y0, y1 = ys, min(ys + hole_h, H)
                    x0, x1 = xs, min(xs + hole_w, W)
                    if z0 < D and y0 < H and x0 < W:
                        mask[z0:z1, y0:y1, x0:x1] = True
        return mask


# ── MixUp 3D ──────────────────────────────────────────────────────────────────

def mixup_3d(
    ct_a: np.ndarray,
    pet_a: np.ndarray,
    label_a: float,
    ct_b: np.ndarray,
    pet_b: np.ndarray,
    label_b: float,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    MixUp two samples.

    Returns:
        mixed_ct, mixed_pet, mixed_label
    """
    lam = np.random.beta(alpha, alpha)
    ct_mix    = lam * ct_a    + (1 - lam) * ct_b
    pet_mix   = lam * pet_a   + (1 - lam) * pet_b
    label_mix = lam * label_a + (1 - lam) * label_b
    return ct_mix, pet_mix, float(label_mix)


# ── Spatial augmentations ─────────────────────────────────────────────────────

def random_flip_3d(
    ct: np.ndarray,
    pet: np.ndarray,
    prob: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random horizontal / vertical / depth flip (axes 1, 2, 3 of C,D,H,W)."""
    for axis in [1, 2, 3]:
        if random.random() < prob:
            ct  = np.flip(ct,  axis=axis).copy()
            pet = np.flip(pet, axis=axis).copy()
    return ct, pet


def random_rotate90_3d(
    ct: np.ndarray,
    pet: np.ndarray,
    prob: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random 90°/180°/270° rotation in the axial (X-Y) plane.

    Input shape: (C, X, Y, Z) = (C, 72, 72, 24).
    We rotate axes (1, 2) = (X, Y), both size 72 → shape is preserved.
    Rotating axes (2, 3) = (Y, Z) would change shape since Y≠Z (72≠24).
    """
    if random.random() < prob:
        k = random.choice([1, 2, 3])
        ct  = np.rot90(ct,  k=k, axes=(1, 2)).copy()
        pet = np.rot90(pet, k=k, axes=(1, 2)).copy()
    return ct, pet


# ── Composed augmentation ─────────────────────────────────────────────────────

class Augment3D:
    """
    Apply GridMask + random flips + random 90° rotations to a pair of
    (ct, pet) volumes. MixUp is handled at the batch level in the
    training loop because it requires two samples.
    """

    def __init__(
        self,
        gridmask_prob: float = 0.5,
        gridmask_ratio: float = 0.4,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.3,
    ):
        self.gridmask = GridMask3D(
            min_hole=4,
            max_hole=12,
            ratio=gridmask_ratio,
            prob=gridmask_prob,
        )
        self.flip_prob   = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(
        self,
        ct: np.ndarray,
        pet: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ct, pet = random_flip_3d(ct, pet, self.flip_prob)
        ct, pet = random_rotate90_3d(ct, pet, self.rotate_prob)
        ct, pet = self.gridmask(ct, pet)
        return ct, pet
