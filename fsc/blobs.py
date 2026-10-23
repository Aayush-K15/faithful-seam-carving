"""fsc.blobs - simple object masks for photos YOLO does not know (coins, cells, parts on a plain background)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation


def blob_labels(img: np.ndarray, min_frac: float = 0.003) -> np.ndarray:
    """
    Label separate blobs: Otsu threshold (polarity chosen so objects are the minority),
    fill holes, split touching blobs with a distance-transform watershed, drop tiny ones.
    Works for objects on a fairly plain background; not a general detector.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    fg = g > filters.threshold_otsu(g)
    if fg.mean() > 0.5:
        fg = ~fg
    fg = ndi.binary_fill_holes(ndi.binary_opening(fg, structure=morphology.disk(2)))
    dist = ndi.distance_transform_edt(fg)
    markers = measure.label(morphology.h_maxima(filters.gaussian(dist, 1), 1.5))
    lab = segmentation.watershed(-dist, markers, mask=fg)
    out = np.zeros(lab.shape, np.int32)
    k = 0
    for r in measure.regionprops(lab):
        if r.area >= min_frac * lab.size:
            k += 1
            out[lab == r.label] = k
    return out
