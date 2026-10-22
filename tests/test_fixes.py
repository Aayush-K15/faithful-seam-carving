import cv2
import numpy as np

from fsc.engine import SeamRecord, VERTICAL
from fsc.fixes import CostKneeStop, line_mask, spectral_residual_saliency


def test_line_mask_finds_long_line_not_noise():
    rng = np.random.default_rng(0)
    img = (rng.random((120, 160)) * 40 + 100).astype(np.uint8)   # mild noise
    cv2.line(img, (10, 60), (150, 60), 255, 2)                     # one long horizontal line
    m = line_mask(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert m.shape == (120, 160) and set(np.unique(m)) <= {0.0, 1.0}
    assert m[57:64, 20:140].mean() > 0.5
    assert m[:40].sum() == 0 and m[90:].sum() == 0


def test_saliency_peaks_on_odd_blob():
    img = np.full((96, 128, 3), 120, np.uint8)
    cv2.circle(img, (90, 40), 10, (20, 20, 230), -1)
    s = spectral_residual_saliency(img)
    assert s.shape == (96, 128) and s.min() >= 0 and s.max() <= 1
    assert s[30:50, 80:100].mean() > 3 * s[70:, :40].mean()


def _hist(costs):
    return [SeamRecord(i, VERTICAL, c, c, 0.0, 100 - i) for i, c in enumerate(costs)]


def test_knee_fires_on_step_change_only():
    flat = [1.0 + 0.05 * np.sin(i) for i in range(80)]
    stop = CostKneeStop(warmup=20, window=10, factor=2.5)
    assert not any(stop(_hist(flat[:t])) for t in range(1, 81))
    jump = flat[:50] + [5.0] * 30
    stop = CostKneeStop(warmup=20, window=10, factor=2.5)
    fired = [t for t in range(1, 81) if stop(_hist(jump[:t]))]
    assert fired and 50 < fired[0] <= 60


def test_knee_warn_only_records():
    stop = CostKneeStop(warmup=5, window=3, factor=2.0, warn_only=True)
    costs = [1.0] * 10 + [10.0] * 5
    assert not any(stop(_hist(costs[:t])) for t in range(1, 16))
    assert stop.triggered_at is not None
