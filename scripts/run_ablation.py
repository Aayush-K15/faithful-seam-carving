"""
Run the ablation study on a YOLO-seg dataset (e.g. COCO128-seg or COCO val2017).

    python scripts/run_ablation.py --data data/coco128-seg --out results

Writes
  results/ablation_rows.csv   one row per (image, width ratio, condition)
  results/summary.csv/.md     pooled object-level rates per (ratio, condition)
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.datasets import load_yolo_seg  # noqa: E402
from fsc.fixes import spectral_residual_saliency  # noqa: E402
from fsc.metrics import image_metrics  # noqa: E402

ARGS = None


def eligible(paths, args):
    out = []
    for img_path in paths:
        s = load_yolo_seg(img_path, label_path(img_path), max_side=args.max_side,
                          min_area_frac=args.min_area)
        if s.n_objects >= args.min_objects:
            out.append(img_path)
    return out


def label_path(img_path: Path) -> Path:
    return img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")


def work(img_path):
    args = ARGS
    s = load_yolo_seg(img_path, label_path(img_path), max_side=args.max_side, min_area_frac=args.min_area)
    H, W = s.image.shape[:2]
    sal = spectral_residual_saliency(s.image)
    rows = []
    for ratio in args.ratios:
        tw = round(W * ratio)
        for cond in args.conditions:
            res = conditions.run(cond, s.image, s.labels, tw)
            r = image_metrics(s.labels, res, in_width=W, in_height=H, saliency=sal)
            refusals = getattr(res, "refusals", {})
            rows.append(dict(image=s.name, width=W, height=H, ratio=ratio, condition=cond,
                             refusals_loss=refusals.get("loss", 0), refusals_merge=refusals.get("merge", 0), **r))
    return rows


SUM_KEYS = ["destroyed", "vanished", "worse_than_scaling", "merged_objects"]
MEAN_KEYS = ["merged_pairs", "aspect_distortion", "raggedness", "carved_fraction", "saliency_kept", "seconds"]


def summarise(rows, ratios, conds):
    out = []
    for ratio in ratios:
        for c in conds:
            rs = [r for r in rows if r["ratio"] == ratio and r["condition"] == c]
            if not rs:
                continue
            n_obj = sum(r["n_objects"] for r in rs)
            d = dict(ratio=ratio, condition=c, label=conditions.LABELS[c], images=len(rs), objects=n_obj)
            for k in SUM_KEYS:
                d[f"{k}_pct"] = 100.0 * sum(r[k] for r in rs) / max(n_obj, 1)
            intact = sum(r["intact_rate"] * r["n_objects"] for r in rs)
            d["intact_pct"] = 100.0 * intact / max(n_obj, 1)
            d["images_with_damage_pct"] = 100.0 * np.mean([(r["destroyed"] + r["vanished"] + r["merged_objects"]) > 0
                                                           for r in rs])
            for k in MEAN_KEYS:
                vals = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and np.isfinite(r[k])]
                d[k] = float(np.mean(vals)) if vals else float("nan")
            d["stopped_early_pct"] = 100.0 * np.mean([bool(r["stopped_early"]) for r in rs])
            out.append(d)
    return out


def write_md(summary, path):
    cols = [("label", "Method", "{}"), ("intact_pct", "Objects intact %", "{:.1f}"),
            ("destroyed_pct", "Destroyed %", "{:.1f}"), ("vanished_pct", "Vanished %", "{:.1f}"),
            ("merged_objects_pct", "Fused %", "{:.1f}"), ("worse_than_scaling_pct", "Worse than scaling %", "{:.1f}"),
            ("images_with_damage_pct", "Images with any damage %", "{:.0f}"),
            ("aspect_distortion", "Aspect distortion", "{:.3f}"), ("raggedness", "Raggedness", "{:.2f}"),
            ("carved_fraction", "Carved %", "{:.0%}"), ("seconds", "Seconds", "{:.2f}")]
    lines = []
    for ratio in sorted({s["ratio"] for s in summary}, reverse=True):
        part = [s for s in summary if s["ratio"] == ratio]
        lines.append(f"\n### Width {ratio:.0%} ({part[0]['images']} images, {part[0]['objects']} objects)\n")
        lines.append("| " + " | ".join(h for _, h, _ in cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for s in part:
            lines.append("| " + " | ".join(f.format(s[k]) for k, _, f in cols) + " |")
    Path(path).write_text("\n".join(lines).lstrip() + "\n")


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco128-seg")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-side", type=int, default=320, help="downscale images for speed (0 = full size)")
    ap.add_argument("--min-objects", type=int, default=3)
    ap.add_argument("--min-area", type=float, default=0.005, help="drop objects below this fraction of the image")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.75, 0.5])
    ap.add_argument("--conditions", nargs="+", default=conditions.ALL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ARGS = ap.parse_args()
    ARGS.max_side = ARGS.max_side or None

    paths = sorted((Path(ARGS.data) / "images" / ARGS.split).glob("*.jpg"))
    paths = eligible(paths, ARGS)
    if ARGS.limit:
        paths = paths[:ARGS.limit]
    print(f"{len(paths)} images with >= {ARGS.min_objects} objects; "
          f"{len(ARGS.conditions)} conditions x {len(ARGS.ratios)} ratios")
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    with Pool(ARGS.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(work, paths), 1):
            rows.extend(rs)
            if i % 5 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} images, {time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: (r["image"], -r["ratio"], conditions.ALL.index(r["condition"])))
    with open(out / "ablation_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarise(rows, ARGS.ratios, ARGS.conditions)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    write_md(summary, out / "summary.md")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
