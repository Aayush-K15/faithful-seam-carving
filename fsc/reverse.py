"""
fsc.reverse - make a resized image reversible, and restore the original from it.

The sender, who has the original, packs a small side file from the resize
result. The receiver, who has only the resized image and that file, rebuilds
the original. Three modes trade size for accuracy:

  exact  where every kept pixel came from + the removed pixels + the
         difference the finishing scale introduced -> the original, bit for bit
  small  where pixels came from + the removed pixels -> carved pixels exact,
         the scaled part approximate (scaling blurs)
  tiny   where pixels came from only -> removed pixels guessed by
         interpolating their neighbours; the scaled part approximate

The side file can travel inside the resized PNG (a private ancillary chunk,
``fsCr``, that ordinary viewers ignore) or as a separate file. It only works
if the resized image reaches the receiver unchanged: a chat app that
re-compresses or downsizes photos breaks it, so send the PNG as a file.

    data = pack(original, result, mode="exact")
    png = embed_png(result.image, data)            # one shareable file
    image, data = extract_png(png)                  # at the receiver
    restored = restore(image, data)
"""

from __future__ import annotations

import hashlib
import json
import lzma
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .engine import CarveResult

MODES = ("exact", "small", "tiny")
MAGIC = b"FSCR1\n"
CHUNK = b"fsCr"   # ancillary, private, reserved bit clear, safe to copy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _digest(img: np.ndarray) -> str:
    a = np.ascontiguousarray(img)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def _upsample(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Separable linear resize (half-pixel centres) in float64, rounded to uint8.
    Pure NumPy element-wise arithmetic, so sender and receiver get identical
    values on any machine; that is what makes the stored residual exact.
    """
    a = img.astype(np.float64)
    for axis, n in ((0, height), (1, width)):
        m = a.shape[axis]
        x = np.clip((np.arange(n) + 0.5) * m / n - 0.5, 0.0, m - 1.0)
        x0 = np.floor(x).astype(np.intp)
        x1 = np.minimum(x0 + 1, m - 1)
        t = x - x0
        shape = [1] * a.ndim
        shape[axis] = n
        t = t.reshape(shape)
        a = np.take(a, x0, axis=axis) * (1.0 - t) + np.take(a, x1, axis=axis) * t
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


def _residual_arrays(r: np.ndarray) -> dict:
    """
    Store the scaling residual compactly: zig-zag it into 0..255 and save it as
    lossless WebP (its predictors suit this noise-like image better than plain
    lzma, about 20% smaller). Falls back to raw int16 when that is not possible.
    """
    z = np.where(r >= 0, 2 * r.astype(np.int32), -2 * r.astype(np.int32) - 1)
    if z.max() < 256 and r.ndim == 3 and r.shape[2] == 3:
        z8 = z.astype(np.uint8)
        ok, buf = cv2.imencode(".webp", z8, [cv2.IMWRITE_WEBP_QUALITY, 101])
        if ok and np.array_equal(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), z8):
            return {"residual_webp": buf.ravel()}
    return {"residual": r.astype(np.int16)}


def _residual_from(arrays: dict) -> Optional[np.ndarray]:
    if "residual_webp" in arrays:
        z = cv2.imdecode(np.asarray(arrays["residual_webp"]), cv2.IMREAD_UNCHANGED).astype(np.int16)
        return np.where(z % 2 == 0, z // 2, -(z + 1) // 2).astype(np.int16)
    return arrays.get("residual")


def _encode(arrays: dict, meta: dict) -> bytes:
    header = dict(meta, arrays=[])
    blobs = []
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        blob = lzma.compress(arr.tobytes(), preset=9 | lzma.PRESET_EXTREME)
        header["arrays"].append(dict(name=name, dtype=arr.dtype.str, shape=list(arr.shape), size=len(blob)))
        blobs.append(blob)
    h = json.dumps(header, separators=(",", ":")).encode()
    return MAGIC + struct.pack(">I", len(h)) + h + b"".join(blobs)


def _decode(data: bytes) -> tuple[dict, dict]:
    if not data.startswith(MAGIC):
        raise ValueError("not a Faithful Seam Carving side file")
    n = struct.unpack(">I", data[len(MAGIC):len(MAGIC) + 4])[0]
    pos = len(MAGIC) + 4
    header = json.loads(data[pos:pos + n])
    pos += n
    arrays = {}
    for a in header.pop("arrays"):
        raw = lzma.decompress(data[pos:pos + a["size"]])
        arrays[a["name"]] = np.frombuffer(raw, dtype=np.dtype(a["dtype"])).reshape(a["shape"])
        pos += a["size"]
    return header, arrays


def _carved(res: CarveResult, original: np.ndarray):
    """Image and flat source indices after carving, before the finishing scale."""
    H, W = original.shape[:2]
    if res.carved_image is not None and res.carved_src_rows is not None:
        img, rows, cols = res.carved_image, res.carved_src_rows, res.carved_src_cols
    elif res.finish_applied in (None, "crop"):          # no scaling: the output maps are exact
        img, rows, cols = res.image, res.src_rows, res.src_cols
    else:
        raise ValueError("this result has no pre-finish source maps; re-run it with the current engine")
    return np.asarray(img), rows.astype(np.int64) * W + cols


# --------------------------------------------------------------------------- #
# Sender
# --------------------------------------------------------------------------- #

def pack(original: np.ndarray, res: CarveResult, mode: str = "exact") -> bytes:
    """Side file that lets someone holding ``res.image`` rebuild ``original``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    original = np.asarray(original)
    if original.dtype != np.uint8 or res.image.dtype != np.uint8:
        raise ValueError("only 8-bit images are supported")
    H, W = original.shape[:2]
    carved, flat = _carved(res, original)
    h, w = carved.shape[:2]
    out = np.asarray(res.image)
    scaled = out.shape[:2] != (h, w)

    # where every carved pixel came from, as row-wise differences (mostly 1s)
    first = flat[:, :1]
    steps = np.diff(flat, axis=1)
    arrays = {"first": first.astype(np.int32), "steps": steps.astype(np.int32)}

    if mode in ("exact", "small"):
        kept = np.zeros(H * W, dtype=bool)
        kept[flat.ravel()] = True
        missing = ~kept.reshape(H, W)
        removed = original[missing]                      # raster order
        per_row = missing.sum(axis=1)
        if per_row.min() == per_row.max() and per_row[0] > 0:
            # vertical seams only: one value per row per seam; consecutive rows of a seam are neighbours,
            # so store differences down each seam (much smaller after compression)
            k = int(per_row[0])
            r = removed.reshape((H, k) + original.shape[2:]).astype(np.int16)
            arrays["removed_delta"] = np.concatenate([r[:1], np.diff(r, axis=0)], axis=0)
        else:
            arrays["removed"] = removed

    if mode == "exact" and scaled:
        approx = _upsample(out, h, w)
        arrays.update(_residual_arrays(carved.astype(np.int16) - approx.astype(np.int16)))
    elif mode == "exact" and not np.array_equal(carved, out):
        raise ValueError("output differs from the carved image although no scaling was applied")

    meta = dict(version=1, mode=mode, original_shape=list(original.shape), carved_shape=[h, w],
                output_shape=list(out.shape), output_sha256=_digest(out))
    return _encode(arrays, meta)


# --------------------------------------------------------------------------- #
# Receiver
# --------------------------------------------------------------------------- #

@dataclass
class Restored:
    image: np.ndarray            # the rebuilt original
    mode: str                    # side-file mode
    exact: bool                  # True if the rebuild is guaranteed bit-exact
    output_matches: bool         # False: the resized image was changed in transit
    guessed: np.ndarray          # (H, W) bool: pixels that were interpolated, not stored


def _fill(img: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill missing pixels by linear interpolation along rows, then columns for whole missing rows."""
    out = img.astype(np.float64)
    hole = missing.copy()
    for axis in (1, 0):
        a = out if axis == 1 else np.swapaxes(out, 0, 1)
        m = hole if axis == 1 else hole.T
        for i in range(a.shape[0]):
            miss = m[i]
            if not miss.any() or miss.all():
                continue
            xs = np.flatnonzero(~miss)
            qs = np.flatnonzero(miss)
            vals = a[i]
            if vals.ndim == 1:
                vals[qs] = np.interp(qs, xs, vals[xs])
            else:
                for c in range(vals.shape[1]):
                    vals[qs, c] = np.interp(qs, xs, vals[xs, c])
            miss[:] = False
        if not hole.any():
            break
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def restore(image: np.ndarray, data: bytes) -> Restored:
    """Rebuild the original from the resized image and its side file."""
    meta, arrays = _decode(data)
    image = np.asarray(image)
    H, W = meta["original_shape"][:2]
    h, w = meta["carved_shape"]
    if list(image.shape) != meta["output_shape"]:
        raise ValueError(f"resized image is {image.shape}, the side file expects {tuple(meta['output_shape'])}: "
                         "it was resized or converted in transit")
    matches = _digest(image) == meta["output_sha256"]

    flat = np.concatenate([arrays["first"].astype(np.int64),
                           arrays["first"].astype(np.int64) + np.cumsum(arrays["steps"], axis=1)], axis=1)
    scaled = image.shape[:2] != (h, w)
    carved = _upsample(image, h, w) if scaled else image.copy()
    residual = _residual_from(arrays)
    if residual is not None:
        carved = (carved.astype(np.int16) + residual).clip(0, 255).astype(np.uint8)

    out = np.zeros(tuple(meta["original_shape"]), dtype=np.uint8)
    flat_out = out.reshape((H * W,) + out.shape[2:])
    # an enlarged image holds some pixels twice; the first copy is the original value
    src, first = np.unique(flat.ravel(), return_index=True)
    flat_out[src] = carved.reshape((h * w,) + carved.shape[2:])[first]
    kept = np.zeros(H * W, dtype=bool)
    kept[src] = True
    missing = ~kept.reshape(H, W)

    if "removed_delta" in arrays:
        flat_out[~kept] = np.cumsum(arrays["removed_delta"], axis=0).reshape((-1,) + out.shape[2:]).astype(np.uint8)
        guessed = np.zeros((H, W), dtype=bool)
    elif "removed" in arrays:
        flat_out[~kept] = arrays["removed"]
        guessed = np.zeros((H, W), dtype=bool)
    else:
        out = _fill(out, missing)
        guessed = missing
    exact = matches and (residual is not None or not scaled) and not guessed.any()
    return Restored(out, meta["mode"], bool(exact), bool(matches), guessed)


# --------------------------------------------------------------------------- #
# PNG container
# --------------------------------------------------------------------------- #

def _chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def embed_png(image: np.ndarray, data: bytes) -> bytes:
    """PNG of ``image`` that also carries the side file; viewers show the image as usual."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode PNG")
    png = buf.tobytes()
    iend = png.rfind(b"IEND") - 4
    return png[:iend] + _chunk(CHUNK, data) + png[iend:]


def extract_png(png: bytes) -> tuple[np.ndarray, Optional[bytes]]:
    """Image and side file (or None) from a PNG made by ``embed_png``."""
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    pos, data = 8, None
    while pos + 8 <= len(png):
        n = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        if kind == CHUNK:
            data = png[pos + 8:pos + 8 + n]
        pos += 12 + n
    return image, data


def side_file_sizes(data: bytes) -> dict:
    """Compressed bytes per component of a side file (for reporting)."""
    n = struct.unpack(">I", data[len(MAGIC):len(MAGIC) + 4])[0]
    header = json.loads(data[len(MAGIC) + 4:len(MAGIC) + 4 + n])
    sizes = {a["name"]: a["size"] for a in header["arrays"]}
    sizes["header"] = len(MAGIC) + 4 + n
    sizes["total"] = len(data)
    return sizes


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (inf if identical)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
