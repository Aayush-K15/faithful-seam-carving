"""
fsc.baselines - non-seam baselines, returned in the same CarveResult form as the
engine so every metric treats all methods identically.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from .engine import CarveResult, best_crop_window, _nearest_resample


def scale(image: np.ndarray, target_width: int, target_height: Optional[int] = None) -> CarveResult:
    """Uniform scaling (INTER_AREA). Every original pixel contributes to the output."""
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    th = H if target_height is None else target_height
    out = cv2.resize(image, (target_width, th), interpolation=cv2.INTER_AREA)
    out = out.reshape((th, target_width) + image.shape[2:])
    rows, cols = np.indices((H, W), dtype=np.int32)
    return CarveResult(out, _nearest_resample(rows, th, target_width),
                       _nearest_resample(cols, th, target_width),
                       np.ones((H, W), dtype=np.int32), [], {}, {}, False, None, "scale",
                       (H, W), time.perf_counter() - t0, np.asarray(image).copy(), rows, cols)


def crop(image: np.ndarray, target_width: int, target_height: Optional[int] = None,
         importance: Optional[np.ndarray] = None) -> CarveResult:
    """Crop to the target size: centre crop, or the window with the most ``importance``."""
    t0 = time.perf_counter()
    H, W = image.shape[:2]
    th = H if target_height is None else target_height
    imp = np.zeros((H, W)) if importance is None else importance
    y0, x0 = best_crop_window(imp, th, target_width)
    rows, cols = np.indices((H, W), dtype=np.int32)
    counts = np.zeros((H, W), dtype=np.int32)
    counts[y0:y0 + th, x0:x0 + target_width] = 1
    sl = (slice(y0, y0 + th), slice(x0, x0 + target_width))
    out = np.ascontiguousarray(image[sl])
    return CarveResult(out, rows[sl].copy(), cols[sl].copy(), counts,
                       [], {}, {}, False, None, "crop", (H, W), time.perf_counter() - t0,
                       out, rows[sl].copy(), cols[sl].copy())
