"""
fsc.datasets - images with instance label maps.

* ``load_yolo_seg``: COCO-derived sets in YOLO segmentation format (one .txt
  per image, one line per instance: ``class x1 y1 x2 y2 ...`` with coordinates
  normalised to [0, 1]). COCO128-seg is the small public set used in the
  examples; the full COCO val2017 converts to the same format.
* ``predict_instances``: masks from a pretrained YOLO segmentation model, for
  your own photos (needs ``pip install ultralytics``; not used in evaluation,
  where ground-truth masks avoid detector error).

A label map is (H, W) int32: 0 = background, 1..N = objects. Where instances
overlap, the smaller one is drawn on top so it is not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence

import cv2
import numpy as np

COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)


@dataclass
class Sample:
    name: str
    image: np.ndarray          # (H, W, 3) BGR uint8
    labels: np.ndarray         # (H, W) int32
    classes: list[int]         # COCO class id of object k at index k-1

    @property
    def n_objects(self) -> int:
        return int(self.labels.max())


def draw_label_map(shape: tuple[int, int], polygons: Sequence[np.ndarray],
                   classes: Sequence[int], min_area_px: float = 0.0) -> tuple[np.ndarray, list[int]]:
    """Rasterise polygons (pixel coords) into a label map, largest first; drop tiny objects."""
    H, W = shape
    order = sorted(range(len(polygons)), key=lambda k: -abs(cv2.contourArea(polygons[k].astype(np.float32))))
    lab = np.zeros((H, W), dtype=np.int32)
    for rank, k in enumerate(order, start=1):
        cv2.fillPoly(lab, [np.rint(polygons[k]).astype(np.int32)], rank)
    area = np.bincount(lab.ravel(), minlength=len(order) + 1)
    out = np.zeros_like(lab)
    kept_classes = []
    for rank, k in enumerate(order, start=1):
        if area[rank] > 0 and area[rank] >= min_area_px:
            kept_classes.append(int(classes[k]))
            out[lab == rank] = len(kept_classes)
    return out, kept_classes


def load_yolo_seg(image_path: str | Path, label_path: str | Path, *, max_side: Optional[int] = None,
                  min_area_frac: float = 0.005, classes: Optional[set[int]] = None) -> Sample:
    """Load one image and its YOLO-seg polygons as a label map (optionally downscaled)."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    if max_side and max(image.shape[:2]) > max_side:
        s = max_side / max(image.shape[:2])
        image = cv2.resize(image, (round(image.shape[1] * s), round(image.shape[0] * s)),
                           interpolation=cv2.INTER_AREA)
    H, W = image.shape[:2]
    polys, cls = [], []
    lp = Path(label_path)
    if lp.exists():
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:          # class + at least 3 points
                continue
            c = int(float(parts[0]))
            if classes is not None and c not in classes:
                continue
            xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2) * [W, H]
            polys.append(xy)
            cls.append(c)
    labels, kept = draw_label_map((H, W), polys, cls, min_area_frac * H * W)
    return Sample(Path(image_path).stem, image, labels, kept)


def iter_yolo_seg(root: str | Path, *, split: str = "train2017", **kw) -> Iterator[Sample]:
    """Iterate a YOLO-seg dataset laid out as root/images/<split>, root/labels/<split>."""
    root = Path(root)
    for img_path in sorted((root / "images" / split).glob("*.jpg")):
        yield load_yolo_seg(img_path, root / "labels" / split / (img_path.stem + ".txt"), **kw)


def predict_instances(image: np.ndarray, model: str = "yolo11n-seg.pt", conf: float = 0.4,
                      min_area_frac: float = 0.005) -> tuple[np.ndarray, list[int]]:
    """Label map from a pretrained YOLO segmentation model (ultralytics)."""
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("predict_instances needs `pip install ultralytics`") from e
    result = YOLO(model).predict(image, conf=conf, verbose=False)[0]
    H, W = image.shape[:2]
    if result.masks is None:
        return np.zeros((H, W), dtype=np.int32), []
    polys = [np.asarray(p, dtype=np.float64) for p in result.masks.xy if len(p) >= 3]
    cls = [int(c) for c, p in zip(result.boxes.cls.tolist(), result.masks.xy) if len(p) >= 3]
    return draw_label_map((H, W), polys, cls, min_area_frac * H * W)
