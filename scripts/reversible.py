"""
Reversible resizing from the command line.

    # sender: resize and write one PNG that also carries the undo data
    python scripts/reversible.py pack photo.jpg --width 50% -o shared.png              # YOLO masks
    python scripts/reversible.py pack coins.png --width 50% --blobs -o shared.png      # blob masks
    python scripts/reversible.py pack photo.jpg --width 50% --mode tiny -o shared.png  # a few KB, approximate

    # receiver: rebuild the original
    python scripts/reversible.py restore shared.png -o restored.png

The PNG opens as the resized image in any viewer. Send it as a file: apps that
re-compress or shrink photos break the undo data.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsc import conditions  # noqa: E402
from fsc.reverse import MODES, embed_png, extract_png, pack, psnr, restore, side_file_sizes  # noqa: E402


def size_arg(text, current):
    if text is None:
        return current
    return round(current * float(text[:-1]) / 100) if text.endswith("%") else int(text)


def cmd_pack(a):
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    if a.max_side and max(img.shape[:2]) > a.max_side:
        k = a.max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    if a.labels:
        labels = cv2.imread(a.labels, cv2.IMREAD_UNCHANGED).astype(np.int32)
        labels = labels[..., 0] if labels.ndim == 3 else labels
    elif a.blobs:
        from fsc.blobs import blob_labels
        labels = blob_labels(img)
    else:
        from fsc.datasets import predict_instances
        labels, _ = predict_instances(img)
    H, W = img.shape[:2]
    tw, th = size_arg(a.width, W), size_arg(a.height, H)
    res = conditions.run(a.method, img, labels, tw, th if th != H else None)
    side = pack(img, res, a.mode)
    Path(a.output).write_bytes(embed_png(res.image, side))
    back = restore(res.image, side)
    q = "bit-exact" if back.exact else f"approximate, PSNR {psnr(back.image, img):.1f} dB"
    print(f"{W}x{H} -> {res.image.shape[1]}x{res.image.shape[0]} | {labels.max()} objects | {a.method} | "
          f"undo data {side_file_sizes(side)['total'] / 1024:.1f} KB ({a.mode}, restores {q}) -> {a.output}")
    if a.max_side:
        print(f"note: the photo was first reduced to {a.max_side} px; restore gives back that image")


def cmd_restore(a):
    image, side = extract_png(Path(a.png).read_bytes())
    if side is None:
        sys.exit("this PNG carries no undo data")
    back = restore(image, side)
    cv2.imwrite(a.output, back.image)
    if not back.output_matches:
        print("warning: the resized image was changed after it was made; result is approximate")
    print(f"{image.shape[1]}x{image.shape[0]} -> {back.image.shape[1]}x{back.image.shape[0]} | mode {back.mode} | "
          f"{'bit-exact' if back.exact else 'approximate'} -> {a.output}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="resize and embed undo data")
    p.add_argument("image")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--width")
    p.add_argument("--height")
    p.add_argument("--method", default="faithful", choices=conditions.ALL)
    p.add_argument("--mode", default="exact", choices=MODES)
    p.add_argument("--labels", help="label PNG (0 = background, 1..N objects)")
    p.add_argument("--blobs", action="store_true", help="simple blob masks instead of YOLO")
    p.add_argument("--max-side", type=int, default=0, help="reduce the photo first (faster); 0 = full size")
    p.set_defaults(fn=cmd_pack)
    r = sub.add_parser("restore", help="rebuild the original from a reversible PNG")
    r.add_argument("png")
    r.add_argument("-o", "--output", required=True)
    r.set_defaults(fn=cmd_restore)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
