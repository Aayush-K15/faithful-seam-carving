"""
Build a paired-comparison user study (the RetargetMe protocol, small scale).

    python scripts/make_user_study.py --images 000000000368 000000000073 ... --ratio 0.5

For every image and every pair of methods, writes one trial image showing the
original on top and the two results side by side (A left, B right, in random
order), plus:
  study/key.csv        trial -> which method is A and which is B (keep hidden)
  study/answers.csv    blank sheet: one row per participant x trial, answer A or B

Analyse with: share of wins per method, and a binomial test per pair.
"""
import argparse
import csv
import itertools
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.datasets import load_yolo_seg  # noqa: E402


def trial_image(original, a, b, gap=16):
    h = max(a.shape[0], b.shape[0])
    row = np.full((h, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    row[:a.shape[0], :a.shape[1]] = a
    row[:b.shape[0], a.shape[1] + gap:] = b
    for text, x in (("A", 8), ("B", a.shape[1] + gap + 8)):
        cv2.putText(row, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(row, text, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    width = max(original.shape[1], row.shape[1])
    pad = lambda im: np.hstack([im, np.full((im.shape[0], width - im.shape[1], 3), 255, np.uint8)])
    return np.vstack([pad(original), np.full((gap, width, 3), 255, np.uint8), pad(row)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/coco128-seg")
    ap.add_argument("--split", default="train2017")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+", default=["scale", "crop", "sc_forward", "faithful"])
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--participants", type=int, default=15)
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="study")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "trials").mkdir(parents=True, exist_ok=True)
    root = Path(args.data)
    key = []
    for name in args.images:
        s = load_yolo_seg(root / "images" / args.split / f"{name}.jpg",
                          root / "labels" / args.split / f"{name}.txt", max_side=args.max_side)
        tw = round(s.image.shape[1] * args.ratio)
        results = {m: conditions.run(m, s.image, s.labels, tw).image for m in args.methods}
        for m1, m2 in itertools.combinations(args.methods, 2):
            a, b = (m1, m2) if rng.random() < 0.5 else (m2, m1)
            tid = f"t{len(key) + 1:03d}"
            cv2.imwrite(str(out / "trials" / f"{tid}.png"), trial_image(s.image, results[a], results[b]))
            key.append(dict(trial=tid, image=name, A=a, B=b))
    rng.shuffle(key)
    with open(out / "key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "image", "A", "B"])
        w.writeheader()
        w.writerows(key)
    with open(out / "answers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant", "trial", "answer (A/B)"])
        for p in range(1, args.participants + 1):
            for k in rng.sample(key, len(key)):
                w.writerow([p, k["trial"], ""])
    print(f"{len(key)} trials in {out / 'trials'}; key in {out / 'key.csv'}")


if __name__ == "__main__":
    main()
