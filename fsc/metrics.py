"""
fsc.metrics - object-integrity metrics for content-aware resizing.

All metrics use the pixel correspondence every method returns (src_rows,
src_cols, source_counts), so they are exact for seam carving and cropping and
nearest-neighbour exact for scaling. Inputs: the instance label map on the
ORIGINAL image (0 = background, 1..N = objects).

Per object i (area A_i in the original):
  kept_i     fraction of the object's original pixels that still appear in the
             output (scaling keeps all of them; seam carving deletes some)
  size_i     object area in the output / A_i
  destroyed  kept_i < 0.5                      (more than half the object deleted)
  vanished   size_i == 0                       (object entirely gone)
  wts        size_i < output/input area ratio  (damaged more than uniform scaling
             would have damaged it; 2% tolerance)
  merged     touches another object in the output but did not in the original
  intact     not destroyed, not vanished, not merged
  aspect     |log(aspect_out / aspect_in)| of the object's bounding box
             (scaling to 50% width gives log 2 = 0.69 for every object)
  ragged     (perimeter^2 / area) out / in  (> 1: outline became more jagged)

Image level: counts, rates, means, and how much of the size change was done
by carving before any finishing step.

Not here, on purpose: SSIM against the original. SSIM needs equal sizes, and
upscaling the result back to compare favours plain scaling.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .engine import CarveResult, VERTICAL, HORIZONTAL
from .integrity import contact_pairs

DESTROYED_KEEP = 0.5
WTS_TOLERANCE = 0.02


def _bbox_aspect(mask: np.ndarray) -> Optional[float]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)


def _compactness(mask: np.ndarray) -> Optional[float]:
    area = float(mask.sum())
    if area < 1:
        return None
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    return perim * perim / area


def per_object(labels: np.ndarray, res: CarveResult) -> dict[str, np.ndarray]:
    """Per-object arrays (index 0 = object 1)."""
    labels = np.asarray(labels, dtype=np.int32)
    n = int(labels.max())
    H, W = labels.shape
    out_lab = labels[res.src_rows, res.src_cols]
    oh, ow = out_lab.shape
    area0 = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.float64)
    area1 = np.bincount(out_lab.ravel(), minlength=n + 1)[1:].astype(np.float64)
    kept_px = np.bincount(labels.ravel(), weights=(res.source_counts > 0).ravel().astype(np.float64),
                          minlength=n + 1)[1:]
    ratio_area = (oh * ow) / float(H * W)

    size = np.divide(area1, area0, out=np.zeros(n), where=area0 > 0)
    kept = np.divide(kept_px, area0, out=np.zeros(n), where=area0 > 0)

    new_pairs = contact_pairs(out_lab) - contact_pairs(labels)
    merged = np.zeros(n, dtype=bool)
    for a, b in new_pairs:
        merged[a - 1] = merged[b - 1] = True

    aspect = np.full(n, np.nan)
    ragged = np.full(n, np.nan)
    for i in range(1, n + 1):
        if area0[i - 1] == 0 or area1[i - 1] == 0:
            continue
        m0, m1 = labels == i, out_lab == i
        a0, a1 = _bbox_aspect(m0), _bbox_aspect(m1)
        aspect[i - 1] = abs(np.log(a1 / a0))
        if area1[i - 1] >= 50:
            c0, c1 = _compactness(m0), _compactness(m1)
            if c0 and c1:
                ragged[i - 1] = c1 / c0

    destroyed = kept < DESTROYED_KEEP
    vanished = area1 == 0
    wts = size < ratio_area * (1.0 - WTS_TOLERANCE)
    return dict(area0=area0, size=size, kept=kept, destroyed=destroyed, vanished=vanished,
                wts=wts, merged=merged, intact=~(destroyed | vanished | merged),
                aspect=aspect, ragged=ragged, ratio_area=np.full(n, ratio_area),
                new_pairs=np.array(len(new_pairs)))


def image_metrics(labels: np.ndarray, res: CarveResult, *, in_width: int, in_height: int,
                  saliency: Optional[np.ndarray] = None) -> dict:
    """One row of the results table for one (image, method, target size)."""
    po = per_object(labels, res)
    n = len(po["kept"])
    oh, ow = res.image.shape[:2]
    requested = (in_width - ow) + (in_height - oh)
    carved = res.seams_removed.get(VERTICAL, 0) + res.seams_removed.get(HORIZONTAL, 0)
    row = dict(
        n_objects=n,
        destroyed=int(po["destroyed"].sum()),
        vanished=int(po["vanished"].sum()),
        worse_than_scaling=int(po["wts"].sum()),
        merged_pairs=int(po["new_pairs"]),
        merged_objects=int(po["merged"].sum()),
        intact_rate=float(po["intact"].mean()) if n else float("nan"),
        min_kept=float(po["kept"].min()) if n else float("nan"),
        mean_size=float(po["size"].mean()) if n else float("nan"),
        aspect_distortion=float(np.nanmean(po["aspect"])) if np.isfinite(po["aspect"]).any() else float("nan"),
        raggedness=float(np.nanmean(po["ragged"])) if np.isfinite(po["ragged"]).any() else float("nan"),
        carved_fraction=(carved / requested) if requested > 0 else float("nan"),
        stopped_early=bool(res.stopped_early),
        stop_reason=res.stop_reason or "",
        finish=res.finish_applied or "",
        seconds=float(res.elapsed_s),
    )
    if saliency is not None:
        s = np.asarray(saliency, dtype=np.float64)
        row["saliency_kept"] = float((s * (res.source_counts > 0)).sum() / max(s.sum(), 1e-12))
    return row
