"""
Resize your own photo with Faithful Seam Carving.

    # masks from a pretrained YOLO segmentation model (pip install ultralytics)
    python scripts/demo.py photo.jpg --width 50% -o out.png
    # or masks from a label PNG you made (0 = background, 1..N = objects)
    python scripts/demo.py photo.jpg --width 50% --labels labels.png -o out.png
    # compare against plain seam carving
    python scripts/demo.py photo.jpg --width 50% --method sc_forward -o plain.png

    # object removal: delete the objects with these label ids, then restore the width
    python scripts/demo.py photo.jpg --remove 2 -o removed.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.removal import remove_objects  # noqa: E402
from fsc.metrics import image_metrics  # noqa: E402
from fsc.viz import object_overlay  # noqa: E402


def parse_size(text, current):
    if text is None:
        return current
    return round(current * float(text[:-1]) / 100) if text.endswith("%") else int(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--width")
    ap.add_argument("--height")
    ap.add_argument("--labels", help="label PNG (0 = background, 1..N objects); default: YOLO masks")
    ap.add_argument("--method", default="faithful", choices=conditions.ALL)
    ap.add_argument("--remove", type=int, nargs="*", help="object ids to remove (object removal mode)")
    ap.add_argument("--overlay", help="also save the result with object outlines")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        ap.error(f"cannot read {args.image}")
    if args.labels:
        labels = cv2.imread(args.labels, cv2.IMREAD_UNCHANGED).astype(np.int32)
        if labels.ndim == 3:
            labels = labels[..., 0]
    else:
        from fsc.datasets import predict_instances
        labels, classes = predict_instances(img)
        print(f"{labels.max()} objects detected")
    H, W = img.shape[:2]

    if args.remove:
        r = remove_objects(img, labels, args.remove)
        cv2.imwrite(args.output, r.image)
        print(f"removed objects {args.remove} with {r.seams} seams "
              f"({'complete' if r.complete else 'INCOMPLETE'}) -> {args.output}")
        return

    tw, th = parse_size(args.width, W), parse_size(args.height, H)
    res = conditions.run(args.method, img, labels, tw, th if th != H else None)
    cv2.imwrite(args.output, res.image)
    m = image_metrics(labels, res, in_width=W, in_height=H)
    print(f"{W}x{H} -> {res.image.shape[1]}x{res.image.shape[0]} | {args.method} | "
          f"destroyed {m['destroyed']} | fused {m['merged_objects']} | carved {m['carved_fraction']:.0%} "
          f"| {m['seconds']:.1f}s")
    if args.overlay:
        cv2.imwrite(args.overlay, object_overlay(res.image, labels[res.src_rows, res.src_cols]))


if __name__ == "__main__":
    main()
