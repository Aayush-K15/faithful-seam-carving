"""
Side-by-side comparison figures.

    # the N images where plain seam carving does the most damage that ours avoids
    python scripts/make_figures.py --auto 4 --ratio 0.5
    # specific images
    python scripts/make_figures.py --images 000000000368 000000000073 --ratio 0.5

Each figure: original | one tile per condition, with object outlines in a
fixed colour per object, so a fused or vanished object is easy to spot.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.datasets import load_yolo_seg  # noqa: E402
from fsc.metrics import image_metrics  # noqa: E402
from fsc.viz import object_overlay, panel_row  # noqa: E402

DEFAULT = ["scale", "sc_forward", "static_mask", "faithful"]


def pick_auto(rows_csv: Path, ratio: float, n: int) -> list[str]:
    rows = list(csv.DictReader(open(rows_csv)))
    by = {}
    for r in rows:
        if abs(float(r["ratio"]) - ratio) > 1e-9:
            continue
        dmg = int(r["destroyed"]) + int(r["vanished"]) + int(r["merged_objects"])
        by.setdefault(r["image"], {})[r["condition"]] = dmg
    scored = [(v.get("sc_forward", 0) + 0.5 * v.get("static_mask", 0) - 2 * v.get("faithful", 0), k)
              for k, v in by.items()]
    return [k for s, k in sorted(scored, reverse=True)[:n] if s > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco128-seg")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--images", nargs="*", default=[])
    ap.add_argument("--auto", type=int, default=0)
    ap.add_argument("--rows", default="results/ablation_rows.csv")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT)
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--scale", type=int, default=2, help="upscale tiles for readability")
    ap.add_argument("--tint", type=float, default=0.3, help="object tint strength (0 = outlines only)")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--tiles", action="store_true", help="also save every tile as its own PNG (for slides/papers)")
    args = ap.parse_args()

    names = list(args.images) + (pick_auto(Path(args.rows), args.ratio, args.auto) if args.auto else [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    for name in names:
        s = load_yolo_seg(root / "images" / args.split / f"{name}.jpg",
                          root / "labels" / args.split / f"{name}.txt", max_side=args.max_side)
        H, W = s.image.shape[:2]
        tw = round(W * args.ratio)
        tiles = [(f"Original ({s.n_objects} objects)", object_overlay(s.image, s.labels, alpha=args.tint))]
        for c in args.conditions:
            res = conditions.run(c, s.image, s.labels, tw)
            m = image_metrics(s.labels, res, in_width=W, in_height=H)
            note = f"{m['destroyed'] + m['vanished']} destroyed, {m['merged_objects']} fused"
            if c == "scale":
                note = "nothing lost, all squashed"
            tiles.append((conditions.LABELS[c], object_overlay(res.image, s.labels[res.src_rows, res.src_cols],
                                                               alpha=args.tint), note))
        if args.tiles:
            tdir = out / "tiles" / f"{name}_{int(args.ratio * 100)}"
            tdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tdir / "original.png"), tiles[0][1])
            for c, t in zip(args.conditions, tiles[1:]):
                cv2.imwrite(str(tdir / f"{c}.png"), t[1])
        path = out / f"{name}_{int(args.ratio * 100)}.png"
        cv2.imwrite(str(path), panel_row(tiles, scale=args.scale, header=72))
        print(path)


if __name__ == "__main__":
    main()
