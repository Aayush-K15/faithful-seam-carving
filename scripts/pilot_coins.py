"""
Pilot on scikit-image's 'coins' photo (24 well-separated objects).

    python scripts/pilot_coins.py [--out results]

Prints the integrity and detail table at 70% and 50% width and saves a labelled figure.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import data, filters, measure, morphology, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.metrics import image_metrics  # noqa: E402
from fsc.viz import object_overlay, panel_row  # noqa: E402

METHODS = ["scale", "sc_forward", "static_mask", "faithful@0.5", "faithful"]


def segment_coins(im):
    elevation = filters.sobel(im)
    markers = np.zeros_like(im, dtype=np.int32)
    markers[im < 30] = 1
    markers[im > 150] = 2
    seg = ndi.binary_fill_holes(segmentation.watershed(elevation, markers) == 2)
    seg = morphology.remove_small_objects(seg, max_size=50)
    return measure.label(seg).astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gray = data.coins()
    img = gray                      # the photo is grayscale; carve it as such
    lab = segment_coins(gray)
    H, W = gray.shape
    print(f"coins: {W}x{H}, {lab.max()} objects")
    for ratio in (0.7, 0.5):
        tw = round(W * ratio)
        print(f"\n== {ratio:.0%} width ==")
        print(f"{'method':14s} {'destroyed':>9s} {'merged':>6s} {'detail':>6s} {'lost':>4s} {'squash':>6s} {'carved':>6s} {'sec':>5s}")
        tiles = [("Original", object_overlay(img, lab))]
        for m in METHODS:
            res = conditions.run(m, img, lab, tw)
            r = image_metrics(lab, res, in_width=W, in_height=H, image=img)
            print(f"{m:14s} {r['destroyed']:9d} {r['merged_pairs']:6d} {r['detail']:6.2f} {r['detail_lost']:4d} "
                  f"{r['aspect_distortion']:6.2f} {r['carved_fraction']:6.0%} {r['seconds']:5.1f}")
            note = f"{r['destroyed']} destroyed, {r['merged_pairs']} fused, detail {r['detail']:.2f}"
            tiles.append((conditions.label(m), object_overlay(res.image, lab[res.src_rows, res.src_cols]), note))
        cv2.imwrite(str(out / f"pilot_coins_{int(ratio * 100)}.png"), panel_row(tiles, scale=2))


if __name__ == "__main__":
    main()
