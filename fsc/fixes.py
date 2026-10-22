"""
fsc.fixes - the three lightweight fixes from the original study plan.

These have prior work (cite it): edge-preserving seam carving (Kiess et al.,
SPIE 2010), saliency-driven resizing (Achanta & Suesstrunk, ICIP 2009) with
spectral-residual saliency (Hou & Zhang, CVPR 2007), and stopping carving
before artifacts appear (Dong et al., SIGGRAPH Asia 2009). In this project they
are ablation components; the new contribution is in fsc.integrity.

Fix 1  line_mask()                    Canny + probabilistic Hough -> mask of long straight lines
Fix 2  spectral_residual_saliency()   classical saliency map in [0, 1]
Fix 3  CostKneeStop                   first-derivative threshold on the cumulative seam cost
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .engine import SeamRecord, to_gray


def _gray_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(to_gray(image) * 255.0), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Fix 1: line protection
# --------------------------------------------------------------------------- #

def line_mask(image: np.ndarray, *, canny_low: int = 50, canny_high: int = 150,
              min_length_frac: float = 0.12, max_gap: int = 4, thickness: int = 3,
              hough_threshold: int = 40) -> np.ndarray:
    """
    Mask (float64 in {0, 1}) of long straight structural lines.

    Canny alone also fires on texture (grass, hair), which would penalise
    everything. Keeping only Hough segments longer than ``min_length_frac`` of
    the shorter image side restricts the penalty to structure: horizons,
    building edges, poles.
    """
    gray = cv2.GaussianBlur(_gray_u8(image), (5, 5), 1.2)
    edges = cv2.Canny(gray, canny_low, canny_high)
    H, W = gray.shape
    min_len = max(10, int(min_length_frac * min(H, W)))
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold,
                           minLineLength=min_len, maxLineGap=max_gap)
    mask = np.zeros((H, W), dtype=np.uint8)
    if segs is not None:
        for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):   # (N,1,4) in OpenCV 4, (N,4) in 5
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, thickness)
    return mask.astype(np.float64)


# --------------------------------------------------------------------------- #
# Fix 2: saliency protection
# --------------------------------------------------------------------------- #

def spectral_residual_saliency(image: np.ndarray, *, width: int = 64,
                               blur_sigma: float = 2.5) -> np.ndarray:
    """
    Spectral-residual saliency (Hou & Zhang, CVPR 2007), in plain NumPy.

    The log-amplitude spectrum minus its local average is the "residual"
    (what is unexpected in the image); transforming it back with the original
    phase gives a map that is high on distinct objects. Computed at ``width``
    pixels wide, as in the paper, then resized. Returns (H, W) in [0, 1].
    """
    gray = to_gray(image)
    H, W = gray.shape
    w = min(width, W)
    h = max(1, round(w * H / W))
    small = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    spec = np.fft.fft2(small)
    log_amp = np.log(np.abs(spec) + 1e-9)
    phase = np.angle(spec)
    residual = log_amp - cv2.blur(log_amp, (3, 3), borderType=cv2.BORDER_REFLECT)
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), blur_sigma)
    sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)
    lo, hi = float(sal.min()), float(sal.max())
    return np.zeros((H, W)) if hi - lo < 1e-12 else (sal - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Fix 3: adaptive stop on the seam-cost curve
# --------------------------------------------------------------------------- #

@dataclass
class CostKneeStop:
    """
    Stop when the seam-cost curve bends upwards.

    The slope of the cumulative cost C(t) is the cost of seam t. After a warm-up
    of ``warmup`` seams, the baseline slope is the median warm-up cost. Carving
    stops when the mean slope over the last ``window`` seams exceeds ``factor``
    times that baseline: the first-derivative threshold that marks the knee.

    Only the image-energy part of each seam's cost is used (``energy_cost``),
    never the protection part: a vertical seam must cross every horizontal
    protected line, so protection adds a constant that would hide the knee.
    Costs are compared within one axis only (seams on one axis have equal length).

    ``warn_only=True`` records where it would have stopped but never stops.
    """
    warmup: int = 20
    window: int = 10
    factor: float = 2.5
    warn_only: bool = False
    triggered_at: Optional[int] = field(default=None, init=False)

    def __call__(self, history: Sequence[SeamRecord]) -> bool:
        axis = history[-1].axis
        costs = [r.energy_cost for r in history if r.axis == axis]
        if len(costs) < self.warmup + self.window:
            return False
        base = float(np.median(costs[:self.warmup]))
        recent = float(np.mean(costs[-self.window:]))
        fired = recent > self.factor * max(base, 1e-6)
        if fired and self.triggered_at is None:
            self.triggered_at = history[-1].step
        return fired and not self.warn_only
