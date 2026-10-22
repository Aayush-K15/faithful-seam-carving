"""
fsc.removal - object removal (Avidan & Shamir 2007), reusing the integrity machinery.

This is a demo of the method, not a contribution: the original seam carving
paper already removes user-marked objects, and Dickman & Diefenbach (2024)
select them with text prompts. Here the objects to delete come from the same
instance label map, and every OTHER object keeps its integrity budget, so
removing one person does not squash the person standing next to them.

How: seams must pass through the target, so every non-target pixel gets a
large constant extra cost (a seam has one pixel per row, so this is the same
as giving target pixels a large negative cost, which the engine does not allow).
Seams are removed until no target pixel is left, then the original size is
restored by seam insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .engine import CarveResult, resize
from .integrity import InstanceIntegrity


@dataclass
class RemovalResult:
    image: np.ndarray        # same size as the input, objects removed
    carved: CarveResult      # the removal step (narrower image + source maps)
    seams: int               # seams it took to remove the objects
    complete: bool           # False if some target pixels survived the seam budget


def remove_objects(image: np.ndarray, labels: np.ndarray, ids: Sequence[int], *,
                   bias: float = 1000.0, restore: bool = True) -> RemovalResult:
    labels = np.asarray(labels, dtype=np.int32)
    target = np.isin(labels, list(ids)) & (labels > 0)
    if not target.any():
        raise ValueError(f"no pixels carry the ids {list(ids)}")
    H, W = labels.shape
    others = np.where(target, 0, labels)
    ctrl = InstanceIntegrity(others, gap_cost=0.0) if others.max() > 0 else None
    bias_map = np.where(target, 0.0, bias)

    def energy(view):
        e = view.gather(bias_map)
        return e if ctrl is None else e + ctrl.energy(view)

    def still_there(view, _mask):
        return bool(view.gather(target).any())

    cols = np.flatnonzero(target.any(axis=0))
    budget = min(W - 1, 2 * (cols.max() - cols.min() + 1))
    carved = resize(image, W - budget, energy_fn=energy, guard_fn=still_there, finish="none")
    seams = len(carved.history)
    complete = not target[carved.src_rows, carved.src_cols].any()
    out = resize(carved.image, W).image if restore else carved.image
    return RemovalResult(out, carved, seams, complete)
