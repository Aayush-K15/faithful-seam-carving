"""
fsc.integrity - integrity-preserving seam carving (the project's contribution).

Seam carving can silently change what a photo shows: an object is carved away,
or the background between two objects is removed until they fuse. This module
adds three things on top of the forward-energy engine, all driven by an
instance label map (0 = background, 1..N = objects) on the ORIGINAL image:

1. Per-object budget (``InstanceIntegrity.energy``, an engine ``energy_fn``).
   Recomputed before every seam. A pixel of object i costs

       lambda * exp(gamma * d_i),   d_i = 1 - (pixels of i left) / (pixels of i at start)

   so seams spread damage across objects instead of eating the same one, and
   an object becomes expensive as it shrinks. gamma = 0 is a static mask.

2. Gap protection (same energy). Background pixels in a gap of at most
   ``gap_px`` pixels between two DIFFERENT objects, measured along the carving
   direction, cost ``gap_cost`` extra, so neighbours are not pushed together.

3. Integrity guard (``InstanceIntegrity.guard``, an engine ``guard_fn``).
   Before a seam is removed it is refused if it would (a) leave any object
   with more than ``max_loss`` of its pixels deleted, or (b) create a contact
   between two objects that did not touch in the original (checked exactly,
   4-connectivity). A refused seam is the cheapest seam available, so every
   other seam is worse: carving stops and the engine finishes by scaling or
   cropping.

max_loss is a dial. At 0.5 every object survives as a separate thing, but
seams still run through objects and ruin their detail (per-object SSIM on
COCO at half width: 0.57, half the objects below 0.5). At 0 (the default)
seams may only take background; carving stops at the first seam that would
touch an object, and scaling finishes the job. That keeps object detail
above plain scaling (0.91 vs 0.87) with less squash (0.46 vs 0.69). Note that
at 0 no object ever loses pixels, so d_i stays 0 and the budget behaves like
a static mask; the budget matters for thresholds above 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .engine import HORIZONTAL, VERTICAL, CarveView


# --------------------------------------------------------------------------- #
# Label-map helpers
# --------------------------------------------------------------------------- #

def contact_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Pairs (a, b), a < b, of different objects that are 4-adjacent somewhere."""
    labels = np.asarray(labels)
    out: set[tuple[int, int]] = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            lo = np.minimum(a[m], b[m]).astype(np.int64)
            hi = np.maximum(a[m], b[m]).astype(np.int64)
            for x, y in np.unique(np.stack([lo, hi], 1), axis=0):
                out.add((int(x), int(y)))
    return out


def _row_gap_mask(labels: np.ndarray, gap_px: int) -> np.ndarray:
    """Background pixels in a run of <= gap_px pixels, along each ROW, between two different objects."""
    H, W = labels.shape
    cols = np.broadcast_to(np.arange(W), (H, W))
    obj = labels > 0
    left = np.maximum.accumulate(np.where(obj, cols, -1), axis=1)
    right = np.flip(np.minimum.accumulate(np.flip(np.where(obj, cols, W), axis=1), axis=1), axis=1)
    rows = np.arange(H)[:, None]
    lab_l = np.where(left >= 0, labels[rows, np.clip(left, 0, W - 1)], 0)
    lab_r = np.where(right < W, labels[rows, np.clip(right, 0, W - 1)], 0)
    gap = right - left - 1
    return (~obj) & (lab_l > 0) & (lab_r > 0) & (lab_l != lab_r) & (gap <= gap_px)


def gap_mask(labels: np.ndarray, gap_px: int, axis: str) -> np.ndarray:
    """
    Thin gaps that a seam on ``axis`` would close: a vertical seam removes one
    pixel per row, so it narrows gaps along rows; a horizontal seam narrows
    gaps along columns.
    """
    if axis == VERTICAL:
        return _row_gap_mask(labels, gap_px)
    return np.ascontiguousarray(_row_gap_mask(np.ascontiguousarray(labels.T), gap_px).T)


def remove_masked(arr: np.ndarray, mask: np.ndarray, axis: str) -> np.ndarray:
    """Remove the one masked pixel per row (vertical) or per column (horizontal)."""
    H, W = mask.shape
    if axis == VERTICAL:
        return arr[~mask].reshape(H, W - 1)
    return np.ascontiguousarray(arr.T[~mask.T].reshape(W, H - 1).T)


def object_mask(labels: np.ndarray) -> np.ndarray:
    """1.0 on every object pixel (the static-mask baseline)."""
    return (np.asarray(labels) > 0).astype(np.float64)


# --------------------------------------------------------------------------- #
# The integrity controller
# --------------------------------------------------------------------------- #

@dataclass
class InstanceIntegrity:
    labels: np.ndarray            # (H, W) int, 0 = background, 1..N = objects, on the ORIGINAL
    lam: float = 1.0              # base cost of an object pixel
    gamma: float = 8.0            # growth of that cost with the fraction already deleted
    gap_px: int = 3               # gaps this thin (px) between different objects are protected
    gap_cost: float = 50.0        # extra cost of a protected gap pixel (0 disables)
    max_loss: float = 0.0         # guard: refuse seams that delete more than this share of an object
    guard_merges: bool = True     # guard: refuse seams that make two objects touch

    # bookkeeping
    refusals: dict = field(default_factory=lambda: {"loss": 0, "merge": 0}, init=False)
    _cache_key: Optional[tuple] = field(default=None, init=False, repr=False)
    _cache: Optional[tuple] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=np.int32)
        if self.labels.ndim != 2 or (self.labels < 0).any():
            raise ValueError("labels must be a 2-D map of non-negative ints")
        self.n = int(self.labels.max())
        self.orig_area = np.bincount(self.labels.ravel(), minlength=self.n + 1).astype(np.float64)
        self.orig_area[0] = 0.0
        self.orig_contacts = contact_pairs(self.labels)

    # current labels and per-object deletion, cached per (step, axis)
    def _current(self, view: CarveView):
        key = (view.step, view.axis, view.shape)
        if key != self._cache_key:
            lab = view.gather(self.labels)
            area = np.bincount(lab.ravel(), minlength=self.n + 1).astype(np.float64)
            deleted = np.zeros(self.n + 1)
            nz = self.orig_area > 0
            deleted[nz] = 1.0 - area[nz] / self.orig_area[nz]
            self._cache_key, self._cache = key, (lab, area, deleted)
        return self._cache

    def energy(self, view: CarveView) -> np.ndarray:
        """engine energy_fn: per-object budget + thin-gap protection."""
        lab, _, deleted = self._current(view)
        per_object = self.lam * np.exp(self.gamma * np.clip(deleted, 0.0, 1.0))
        per_object[0] = 0.0
        cost = per_object[lab]
        if self.gap_cost > 0 and self.gap_px > 0:
            cost = cost + self.gap_cost * gap_mask(lab, self.gap_px, view.axis)
        return cost

    def guard(self, view: CarveView, mask: np.ndarray) -> bool:
        """engine guard_fn: False if removing this seam breaks an object or fuses two."""
        lab, area, _ = self._current(view)
        removed = np.bincount(lab[mask], minlength=self.n + 1).astype(np.float64)
        nz = self.orig_area > 0
        after = np.zeros(self.n + 1)
        after[nz] = 1.0 - (area[nz] - removed[nz]) / self.orig_area[nz]
        if (after[nz] > self.max_loss + 1e-12).any():
            self.refusals["loss"] += 1
            return False
        if self.guard_merges and self.n > 1:
            new_lab = remove_masked(lab, mask, view.axis)
            if contact_pairs(new_lab) - self.orig_contacts:
                self.refusals["merge"] += 1
                return False
        return True
