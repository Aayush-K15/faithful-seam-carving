"""
Figures for the paper (paper/figs/).

    python scripts/paper_figures.py

coins.pdf  - the coins photo at half width: original, scaling, seam carving, static mask, ours
coco.pdf   - two COCO128-seg photos at half width: original, scaling, seam carving, static mask, ours
dial.pdf   - object detail vs squash at half width for every method (from results/summary.csv)

Prints the per-tile numbers used in the captions.
"""
import csv
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skimage import data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fsc import conditions  # noqa: E402
from fsc.datasets import load_yolo_seg  # noqa: E402
from fsc.metrics import image_metrics  # noqa: E402
from fsc.viz import object_overlay  # noqa: E402
from pilot_coins import segment_coins  # noqa: E402

OUT = ROOT / "paper" / "figs"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#7a7974"
SHORT = {"scale": "Scaling", "sc_forward": "Seam carving", "static_mask": "Static mask", "faithful": "Ours"}


def rgb(img):
    return cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)


def note_for(m, method):
    crushed = m["destroyed"] + m["vanished"]
    return f"{crushed} crushed, {m['merged_objects']} fused\ndetail {m['detail']:.2f}"


def row_tiles(img, lab, methods, ratio=0.5, tint=0.3):
    H, W = lab.shape
    tw = round(W * ratio)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tiles = [("Original", rgb(object_overlay(img, lab, alpha=tint)), f"{lab.max()} objects")]
    for c in methods:
        res = conditions.run(c, img, lab, tw)
        m = image_metrics(lab, res, in_width=W, in_height=H, image=gray)
        print(f"  {c:12s} crushed {m['destroyed'] + m['vanished']:2d} fused {m['merged_objects']:2d} "
              f"pairs {m['merged_pairs']:2d} detail {m['detail']:.3f} squash {m['aspect_distortion']:.3f} "
              f"carved {m['carved_fraction']:.0%}")
        tiles.append((SHORT[c], rgb(object_overlay(res.image, lab[res.src_rows, res.src_cols], alpha=tint)),
                      note_for(m, c)))
    return tiles


def draw_rows(rows, path, width_in):
    W0 = rows[0][0][1].shape[1]
    ratios = [t[1].shape[1] / W0 for t in rows[0]]
    nrows = len(rows)
    h_in = width_in * rows[0][0][1].shape[0] / (W0 * sum(ratios)) * nrows * 1.0 + 0.42 * nrows
    fig, axes = plt.subplots(nrows, len(rows[0]), figsize=(width_in, h_in),
                             gridspec_kw={"width_ratios": ratios, "wspace": 0.04, "hspace": 0.28},
                             squeeze=False)
    for r, tiles in enumerate(rows):
        for c, (title, im, note) in enumerate(tiles):
            ax = axes[r][c]
            ax.imshow(im, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7.5, color=INK, pad=3, fontweight="bold")
            ax.set_xlabel(note, fontsize=6.5, color=INK2, labelpad=2, linespacing=1.1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def coins_figure():
    print("coins (50% width):")
    gray = data.coins()
    lab = segment_coins(gray)
    draw_rows([row_tiles(gray, lab, ["scale", "sc_forward", "static_mask", "faithful"])], OUT / "coins.pdf", 7.16)


def coco_figure(names=("000000000438", "000000000612")):
    root = ROOT / "data" / "coco128-seg"
    rows = []
    for name in names:
        s = load_yolo_seg(root / "images" / "train2017" / f"{name}.jpg",
                          root / "labels" / "train2017" / f"{name}.txt", max_side=320)
        print(f"COCO {name} ({s.image.shape[1]}x{s.image.shape[0]}, {s.n_objects} objects, 50% width):")
        rows.append(row_tiles(s.image, s.labels, ["scale", "sc_forward", "static_mask", "faithful"]))
    draw_rows(rows, OUT / "coco.pdf", 7.16)


def dial_figure():
    rows = {r["condition"]: r for r in csv.DictReader(open(ROOT / "results" / "summary.csv"))
            if abs(float(r["ratio"]) - 0.5) < 1e-9}
    pt = lambda c: (float(rows[c]["aspect_distortion"]), float(rows[c]["detail"]))  # noqa: E731
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.grid(True, color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, width=0.5, length=2)

    sweep = ["faithful@0.5", "faithful@0.2", "faithful@0.1", "faithful@0.05", "faithful"]
    xs, ys = zip(*[pt(c) for c in sweep])
    ax.plot(xs, ys, "-", color=BLUE, linewidth=1.5, zorder=2)
    ax.scatter(xs, ys, s=22, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    offs = {"faithful@0.5": (-5, 2, "right"), "faithful": (-5, 2, "right")}
    for c, lab in zip(sweep, ["50%", "20%", "10%", "5%", "0% (default)"]):
        x, y = pt(c)
        dx, dy, ha = offs.get(c, (5, -3, "left"))
        ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK, ha=ha)
    ax.text(0.205, 0.80, "Ours, guard dial\nfrom 50% to 0%", color=BLUE, fontsize=6.5, fontweight="bold")

    others = [("scale", "Scaling", GREY, (-4, 4), "right"),
              ("sc_forward", "Seam carving", ORANGE, (5, -2), "left"),
              ("fixes_all", "+ lines, saliency, knee", ORANGE, (5, -2), "left"),
              ("static_mask", "Static mask", AQUA, (5, -2), "left"),
              ("budget_gap", "Budget + gaps", AQUA, (-2, -10), "left")]
    for c, lab, col, off, ha in others:
        x, y = pt(c)
        ax.scatter([x], [y], s=22, color=col, edgecolor="white", linewidth=0.8, zorder=3,
                   marker="s" if col == ORANGE else ("D" if col == AQUA else "o"))
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, color=INK, ha=ha)
    ax.set_xlabel("Squash: mean |log aspect change| (lower is better)", color=INK2)
    ax.set_ylabel("Object detail, SSIM (higher is better)", color=INK2)
    ax.set_xlim(0.18, 0.78)
    ax.set_ylim(0.40, 0.96)
    fig.savefig(OUT / "dial.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    coins_figure()
    coco_figure()
    dial_figure()
    print("wrote", *sorted(p.name for p in OUT.iterdir()))
