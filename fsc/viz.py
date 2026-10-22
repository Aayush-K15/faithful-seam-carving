"""fsc.viz - figures: object overlays and labelled side-by-side panels."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(3)
    return (rng.random((max(n, 1) + 1, 3)) * 200 + 55).astype(np.uint8)


def object_overlay(image: np.ndarray, labels: np.ndarray, *, alpha: float = 0.5,
                   outline: bool = True) -> np.ndarray:
    """Tint each object with its own colour (same colour before and after) and draw its outline."""
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    out = img.copy()
    pal = _palette(int(labels.max()))
    m = labels > 0
    out[m] = ((1 - alpha) * out[m] + alpha * pal[labels[m]]).astype(np.uint8)
    if outline:
        for i in range(1, int(labels.max()) + 1):
            mask = (labels == i).astype(np.uint8)
            if mask.any():
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(out, cs, -1, tuple(int(v) for v in pal[i] // 2), 1)
    return out


def panel_row(tiles: Sequence[tuple], *, scale: int = 1, gap: int = 12, header: int = 64) -> np.ndarray:
    """
    tiles: (title, image) or (title, image, note). Images may differ in width but
    share a height. Returns one BGR image with a white header above each tile.
    """
    cols = []
    for t in tiles:
        title, img = t[0], t[1]
        note = t[2] if len(t) > 2 else ""
        if scale != 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        hdr = np.full((header, img.shape[1], 3), 255, np.uint8)
        fs = header / 110.0
        cv2.putText(hdr, title, (6, int(header * 0.42)), cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 25), 2, cv2.LINE_AA)
        if note:
            bad = any(w in note for w in ("destroyed", "fused")) and not note.startswith("0 destroyed, 0")
            color = (40, 40, 170) if bad else (50, 110, 40)
            cv2.putText(hdr, note, (6, int(header * 0.86)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, color, 2,
                        cv2.LINE_AA)
        cols.append(np.vstack([hdr, img]))
    h = max(c.shape[0] for c in cols)
    cols = [np.vstack([c, np.full((h - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cols]
    row = cols[0]
    for c in cols[1:]:
        row = np.hstack([row, np.full((h, gap, 3), 255, np.uint8), c])
    return row
