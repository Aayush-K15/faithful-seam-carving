"""
fsc.engine - seam-carving core engine.

What it implements
------------------
* Forward energy (Rubinstein, Shamir & Avidan, SIGGRAPH 2008) and, for the
  "vanilla seam carving" baseline, the backward gradient energy of Avidan &
  Shamir (SIGGRAPH 2007). Both run through the same dynamic program.
* Seam removal, and enlargement by inserting k seams at once (Avidan & Shamir
  2007): the k seams are found by removing them one by one from a scratch copy,
  which makes them pixel-disjoint in the original, and are then all duplicated.
* Dual-axis resizing: a horizontal seam is a vertical seam of the transposed
  image. When both dimensions shrink, the next seam's axis is chosen greedily
  by cost or by a fixed rule.

Hooks used by the rest of the study (the engine is identical in every condition)
--------------------------------------------------------------------------------
* ``pixel_energy``  static (H, W) extra cost P(i, j), computed once on the
  original and carried with the pixels (line and saliency protection).
* ``energy_fn``     dynamic extra cost, recomputed before EVERY seam from the
  current state (the per-object integrity budget needs this: an object's cost
  depends on how much of it is already gone).
* ``guard_fn``      called with the chosen seam before it is removed; returning
  False stops carving (the integrity-triggered stop).
* ``stop_fn``       called after every removed seam with the cost history
  (the cost-knee adaptive stop).
* ``finish``        when carving stops early, the rest of the size change is
  done by uniform scaling ("scale") or by an importance-guided crop ("crop"),
  so every condition produces exactly the requested size.
* Source maps: every output pixel records the (row, col) it came from in the
  original, and ``source_counts`` records how often each original pixel
  survives (0 = removed, 1 = kept, 2+ = duplicated). All metrics use these.

Conventions
-----------
* Images are arrays as returned by ``cv2.imread``: (H, W, 3) BGR or (H, W)
  grayscale. Integer dtypes are scaled by their max value; float images are
  assumed to be in [0, 1].
* Energy is computed on grayscale intensity in [0, 1]; one forward-energy term
  is in [0, 2]. Scale extra costs against that (1.0 = crossing one full-contrast
  edge).
* A vertical seam has one pixel per row; removing it reduces the width by 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence

import cv2
import numpy as np

VERTICAL = "vertical"      # runs top to bottom; removing it reduces the width
HORIZONTAL = "horizontal"  # runs left to right; removing it reduces the height

ENERGIES = ("forward", "backward")
ORDERS = ("greedy", "width_first", "height_first", "alternate")
FINISHES = ("scale", "crop", "none")

# ITU-R BT.601 luma weights, in OpenCV's BGR channel order.
_BGR_LUMA = np.array([0.114, 0.587, 0.299])


# =========================================================================== #
# Energy
# =========================================================================== #

def to_gray(image: np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] as float64 with shape (H, W)."""
    img = np.asarray(image)
    scale = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
    if img.ndim == 2:
        gray = img.astype(np.float64)
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[..., 0].astype(np.float64)
    elif img.ndim == 3 and img.shape[2] in (3, 4):   # BGR or BGRA (alpha ignored)
        gray = img[..., :3].astype(np.float64) @ _BGR_LUMA
    else:
        raise ValueError(f"expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got {img.shape}")
    return gray / scale


def forward_energy_terms(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rubinstein et al. (2008) forward-energy costs for a vertical seam.

    Removing pixel (i, j) makes (i, j-1) and (i, j+1) neighbours. Depending on
    which pixel the seam used in row i-1, one new vertical neighbour pair also
    appears. The cost of each case is the intensity jump it creates:

        C_U(i,j) = |I(i,j+1) - I(i,j-1)|                            parent (i-1, j)
        C_L(i,j) = |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j-1)|    parent (i-1, j-1)
        C_R(i,j) = |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j+1)|    parent (i-1, j+1)

    Border handling: indices outside the image are clamped. Taken literally,
    removing a border pixel creates no new horizontal pair, which makes the
    outermost column a zero-cost seam and turns carving into cropping. Clamping
    charges the one-sided jump |I(i,1) - I(i,0)| instead.

    Returns (C_U, C_L, C_R), each (H, W) float64.
    """
    p = np.pad(gray, ((1, 0), (1, 1)), mode="edge")
    left = p[1:, :-2]     # I(i, j-1)
    right = p[1:, 2:]     # I(i, j+1)
    up = p[:-1, 1:-1]     # I(i-1, j)
    c_u = np.abs(right - left)
    c_l = c_u + np.abs(up - left)
    c_r = c_u + np.abs(up - right)
    return c_u, c_l, c_r


def backward_energy_terms(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avidan & Shamir (2007) e1 energy |dI/dx| + |dI/dy| (central differences),
    returned as three identical terms so the same dynamic program applies.
    """
    e = np.zeros_like(gray)
    if gray.shape[1] > 1:
        e += np.abs(np.gradient(gray, axis=1))
    if gray.shape[0] > 1:
        e += np.abs(np.gradient(gray, axis=0))
    return e, e, e


def energy_terms(gray: np.ndarray, energy: str = "forward"):
    if energy == "forward":
        return forward_energy_terms(gray)
    if energy == "backward":
        return backward_energy_terms(gray)
    raise ValueError(f"energy must be one of {ENERGIES}, got {energy!r}")


# =========================================================================== #
# Dynamic program and seam extraction
# =========================================================================== #

class Seam(NamedTuple):
    indices: np.ndarray   # (H,) column of the seam in each row
    cost: float           # total: energy_cost + pixel_cost (the DP minimum)
    energy_cost: float    # forward (or backward) energy part, summed along the seam
    pixel_cost: float     # extra-cost part (static + dynamic), summed along the seam


def cumulative_energy(gray: np.ndarray,
                      pixel_energy: Optional[np.ndarray] = None,
                      energy: str = "forward"):
    """
    Minimal cumulative cost M for vertical seams.

        M(0, j) = C_U(0, j) + P(0, j)
        M(i, j) = P(i, j) + min( M(i-1, j-1) + C_L(i, j),
                                 M(i-1, j  ) + C_U(i, j),
                                 M(i-1, j+1) + C_R(i, j) )

    Vectorised across each row; only the H rows are a Python loop.
    Returns (M, back, terms); back[i, j] in {-1, 0, +1} is the column offset of
    the best parent in row i-1. Ties go to the straight-up parent.
    """
    c_u, c_l, c_r = terms = energy_terms(gray, energy)
    H, W = gray.shape
    M = np.empty((H, W), dtype=np.float64)
    back = np.zeros((H, W), dtype=np.int8)

    M[0] = c_u[0] if pixel_energy is None else c_u[0] + pixel_energy[0]
    shifted = np.empty(W, dtype=np.float64)
    for i in range(1, H):
        prev = M[i - 1]
        row = prev + c_u[i]                         # parent (i-1, j)

        shifted[0] = np.inf                         # parent (i-1, j-1)
        shifted[1:] = prev[:-1]
        cand = shifted + c_l[i]
        take = cand < row
        row[take] = cand[take]
        back[i, take] = -1

        shifted[-1] = np.inf                        # parent (i-1, j+1)
        shifted[:-1] = prev[1:]
        cand = shifted + c_r[i]
        take = cand < row
        row[take] = cand[take]
        back[i, take] = 1

        if pixel_energy is not None:
            row += pixel_energy[i]
        M[i] = row
    return M, back, terms


def find_vertical_seam(gray: np.ndarray,
                       pixel_energy: Optional[np.ndarray] = None,
                       energy: str = "forward") -> Seam:
    """Optimal vertical seam, with its cost split into energy and pixel_energy parts."""
    M, back, (c_u, c_l, c_r) = cumulative_energy(gray, pixel_energy, energy)
    H = gray.shape[0]

    seam = np.empty(H, dtype=np.intp)
    j = int(np.argmin(M[-1]))
    seam[-1] = j
    for i in range(H - 1, 0, -1):
        j += int(back[i, j])
        seam[i - 1] = j

    rows = np.arange(1, H)
    cols = seam[1:]
    step = seam[:-1] - seam[1:]                     # parent offset per row
    used = np.where(step == -1, c_l[rows, cols],
                    np.where(step == 0, c_u[rows, cols], c_r[rows, cols]))
    energy_cost = float(c_u[0, seam[0]] + used.sum())
    pixel_cost = 0.0 if pixel_energy is None else float(pixel_energy[np.arange(H), seam].sum())
    return Seam(seam, float(M[-1, seam[-1]]), energy_cost, pixel_cost)


def remove_vertical_seam(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Delete one pixel per row. Works for (H, W) and (H, W, C) arrays of any dtype."""
    H, W = arr.shape[:2]
    keep = np.ones((H, W), dtype=bool)
    keep[np.arange(H), seam] = False
    return arr[keep].reshape((H, W - 1) + arr.shape[2:])


def _remove_seam(arr: np.ndarray, seam: np.ndarray, axis: str) -> np.ndarray:
    if axis == VERTICAL:
        return remove_vertical_seam(arr, seam)
    flipped = remove_vertical_seam(np.swapaxes(arr, 0, 1), seam)
    return np.ascontiguousarray(np.swapaxes(flipped, 0, 1))


def seam_mask(indices: np.ndarray, axis: str, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) bool mask of the pixels a seam would remove, in image orientation."""
    mask = np.zeros(shape, dtype=bool)
    if axis == VERTICAL:
        mask[np.arange(shape[0]), indices] = True
    else:
        mask[indices, np.arange(shape[1])] = True
    return mask


# =========================================================================== #
# What the dynamic hooks see
# =========================================================================== #

@dataclass(frozen=True)
class CarveView:
    """Read-only snapshot passed to ``energy_fn`` and ``guard_fn`` before each seam."""
    gray: np.ndarray       # (H, W) current luminance
    src_rows: np.ndarray   # (H, W) original row of each current pixel
    src_cols: np.ndarray   # (H, W) original column of each current pixel
    axis: str              # axis of the seam about to be found / removed
    step: int              # number of seams removed so far

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape

    def gather(self, original: np.ndarray) -> np.ndarray:
        """Map an array defined on the ORIGINAL image to the current carved image."""
        return original[self.src_rows, self.src_cols]


EnergyFn = Callable[[CarveView], Optional[np.ndarray]]
GuardFn = Callable[[CarveView, np.ndarray], bool]


# =========================================================================== #
# Carving state: every per-pixel array that must move with the pixels
# =========================================================================== #

@dataclass
class _State:
    image: np.ndarray                     # (H, W[, C]) in the caller's dtype
    gray: np.ndarray                      # (H, W) float64 luminance in [0, 1]
    src_rows: np.ndarray                  # (H, W) int32
    src_cols: np.ndarray                  # (H, W) int32
    pixel_energy: Optional[np.ndarray] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.gray.shape

    def _map(self, fn) -> "_State":
        return _State(**{f.name: (None if getattr(self, f.name) is None else fn(getattr(self, f.name)))
                         for f in dataclasses.fields(self)})

    def transpose(self) -> "_State":
        return self._map(lambda a: np.ascontiguousarray(np.swapaxes(a, 0, 1)))

    def remove(self, seam: np.ndarray, axis: str) -> "_State":
        return self._map(lambda a: _remove_seam(a, seam, axis))

    def crop(self, y0: int, x0: int, h: int, w: int) -> "_State":
        return self._map(lambda a: np.ascontiguousarray(a[y0:y0 + h, x0:x0 + w]))

    def view(self, axis: str, step: int) -> CarveView:
        return CarveView(self.gray, self.src_rows, self.src_cols, axis, step)

    def best_seam(self, axis: str, energy: str, extra: Optional[np.ndarray] = None) -> Seam:
        pix = self.pixel_energy
        if extra is not None:
            pix = extra if pix is None else pix + extra
        if axis == VERTICAL:
            return find_vertical_seam(self.gray, pix, energy)
        gray_t = np.ascontiguousarray(self.gray.T)
        pix_t = None if pix is None else np.ascontiguousarray(pix.T)
        return find_vertical_seam(gray_t, pix_t, energy)


def _check_extra(extra, shape, what: str) -> Optional[np.ndarray]:
    if extra is None:
        return None
    extra = np.asarray(extra, dtype=np.float64)
    if extra.shape != shape:
        raise ValueError(f"{what} must return shape {shape}, got {extra.shape}")
    if not np.isfinite(extra).all() or (extra < 0).any():
        raise ValueError(f"{what} must return finite, non-negative costs")
    return extra


# =========================================================================== #
# Removal phase
# =========================================================================== #

@dataclass(frozen=True)
class SeamRecord:
    """One removed seam. ``history`` is what the adaptive-stop rule reads."""
    step: int            # 0-based index over all removed seams
    axis: str            # VERTICAL or HORIZONTAL
    cost: float          # total DP cost = energy_cost + pixel_cost
    energy_cost: float   # forward/backward energy part only
    pixel_cost: float    # protection part only (static + dynamic)
    size_before: int     # width (vertical) or height (horizontal) before removal


StopFn = Callable[[Sequence[SeamRecord]], bool]


def _remove_seams(state: _State, n_vertical: int, n_horizontal: int, *, order: str,
                  energy: str, energy_fn: Optional[EnergyFn], guard_fn: Optional[GuardFn],
                  stop_fn: Optional[StopFn],
                  history: list[SeamRecord]) -> tuple[_State, Optional[str]]:
    """Remove seams until both counts reach zero, the guard refuses, or stop_fn fires."""

    def seam_for(axis: str) -> Seam:
        extra = None
        if energy_fn is not None:
            extra = _check_extra(energy_fn(state.view(axis, len(history))), state.shape, "energy_fn")
        return state.best_seam(axis, energy, extra)

    turn = VERTICAL
    while n_vertical > 0 or n_horizontal > 0:
        seam = None
        if n_vertical and n_horizontal:
            if order == "greedy":
                sv, sh = seam_for(VERTICAL), seam_for(HORIZONTAL)
                axis, seam = (VERTICAL, sv) if sv.cost <= sh.cost else (HORIZONTAL, sh)
            elif order == "width_first":
                axis = VERTICAL
            elif order == "height_first":
                axis = HORIZONTAL
            else:  # alternate
                axis = turn
                turn = HORIZONTAL if turn == VERTICAL else VERTICAL
        else:
            axis = VERTICAL if n_vertical else HORIZONTAL
        if seam is None:
            seam = seam_for(axis)

        if guard_fn is not None:
            mask = seam_mask(seam.indices, axis, state.shape)
            if not guard_fn(state.view(axis, len(history)), mask):
                return state, "guard"

        size_before = state.shape[1] if axis == VERTICAL else state.shape[0]
        state = state.remove(seam.indices, axis)
        history.append(SeamRecord(len(history), axis, seam.cost, seam.energy_cost,
                                  seam.pixel_cost, size_before))
        if axis == VERTICAL:
            n_vertical -= 1
        else:
            n_horizontal -= 1

        if stop_fn is not None and stop_fn(history) and (n_vertical or n_horizontal):
            return state, "stop_fn"
    return state, None


# =========================================================================== #
# Insertion phase
# =========================================================================== #

def _midpoint(a: np.ndarray, b: np.ndarray, dtype) -> np.ndarray:
    m = (a.astype(np.float64) + b) / 2.0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        m = np.clip(np.rint(m), info.min, info.max)
    return m.astype(dtype)


def _expand(arr: np.ndarray, dup: np.ndarray, interpolate: bool = False) -> np.ndarray:
    """
    Insert one new pixel right after every pixel marked in ``dup`` (H, W bool).
    Every row must contain the same number k of marked pixels. With
    ``interpolate`` the new pixel is the mean of the seam pixel and its right
    neighbour; otherwise it is a copy (used for masks and source maps).
    """
    H, W = dup.shape
    k = int(dup[0].sum())
    tail = arr.shape[2:]
    flat = arr.reshape((H * W,) + tail)
    marked = dup.ravel()
    counts = 1 + marked.astype(np.intp)
    out = np.repeat(flat, counts, axis=0)
    if interpolate:
        right = np.concatenate([arr[:, 1:], arr[:, -1:]], axis=1).reshape((H * W,) + tail)
        inserted_at = np.cumsum(counts)[marked] - 1
        out[inserted_at] = _midpoint(flat[marked], right[marked], arr.dtype)
    return out.reshape((H, W + k) + tail)


def _insert_vertical_seams_once(state: _State, k: int, energy: str) -> _State:
    """Simultaneously insert k vertical seams (k <= current width)."""
    H, W = state.shape
    gray, pix = state.gray, state.pixel_energy
    local_cols = np.broadcast_to(np.arange(W), (H, W)).copy()
    dup = np.zeros((H, W), dtype=bool)
    rows = np.arange(H)
    for _ in range(k):
        seam = find_vertical_seam(gray, pix, energy).indices
        dup[rows, local_cols[rows, seam]] = True
        gray = remove_vertical_seam(gray, seam)
        local_cols = remove_vertical_seam(local_cols, seam)
        if pix is not None:
            pix = remove_vertical_seam(pix, seam)
    assert (dup.sum(axis=1) == k).all(), "inserted seams must be pixel-disjoint"

    image = _expand(state.image, dup, interpolate=True)
    return _State(
        image=image,
        gray=to_gray(image),
        src_rows=_expand(state.src_rows, dup),
        src_cols=_expand(state.src_cols, dup),
        pixel_energy=None if state.pixel_energy is None else _expand(state.pixel_energy, dup),
    )


def _insert_seams(state: _State, k: int, axis: str, energy: str, max_fraction: float) -> _State:
    """
    Insert k seams along ``axis`` in passes of at most ``max_fraction`` of the
    current size. Seams are chosen with the removal energy and the static
    pixel_energy only (energy_fn/guard_fn apply to removal).
    """
    if axis == HORIZONTAL:
        return _insert_seams(state.transpose(), k, VERTICAL, energy, max_fraction).transpose()
    while k > 0:
        step = min(k, max(1, int(max_fraction * state.shape[1])))
        state = _insert_vertical_seams_once(state, step, energy)
        k -= step
    return state


# =========================================================================== #
# Finishing crop
# =========================================================================== #

def best_crop_window(importance: np.ndarray, h: int, w: int) -> tuple[int, int]:
    """
    Top-left (y, x) of the h x w window with the largest total importance.
    Ties (e.g. all-zero importance) go to the window closest to the centre.
    """
    H, W = importance.shape
    ii = np.pad(np.asarray(importance, dtype=np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    ys = np.abs(np.arange(sums.shape[0]) - (H - h) / 2.0)
    xs = np.abs(np.arange(sums.shape[1]) - (W - w) / 2.0)
    scale = 1.0 + float(np.abs(sums).max())
    score = sums - 1e-9 * scale * (ys[:, None] + xs[None, :])
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(y), int(x)


# =========================================================================== #
# Public entry point
# =========================================================================== #

@dataclass
class CarveResult:
    image: np.ndarray              # output at the requested size
    src_rows: np.ndarray           # (H', W') original row of each output pixel*
    src_cols: np.ndarray           # (H', W') original column of each output pixel*
    source_counts: np.ndarray      # (H, W) per original pixel: 0 removed, 1 kept, 2+ duplicated
    history: list[SeamRecord]      # every removed seam, in order
    seams_removed: dict[str, int]
    seams_inserted: dict[str, int]
    stopped_early: bool            # carving stopped before all seams were removed
    stop_reason: Optional[str]     # "guard", "stop_fn" or None
    finish_applied: Optional[str]  # "scale", "crop" or None
    carved_shape: tuple[int, int]  # size after seams, before any finishing step
    elapsed_s: float
    # The image and exact source maps after carving (and any crop) but before
    # the finishing scale; used to make a result reversible (fsc.reverse).
    carved_image: Optional[np.ndarray] = None
    carved_src_rows: Optional[np.ndarray] = None
    carved_src_cols: Optional[np.ndarray] = None
    # * Exact unless the finishing scale was applied (then nearest-neighbour
    #   resampled). source_counts never includes the finishing scale (scaling
    #   keeps every pixel's content); it does include a finishing crop.

    @property
    def costs(self) -> np.ndarray:
        return np.array([r.cost for r in self.history])

    @property
    def retained_mask(self) -> np.ndarray:
        return self.source_counts > 0


def _nearest_resample(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    r = np.minimum(((np.arange(height) + 0.5) * arr.shape[0] / height).astype(np.intp), arr.shape[0] - 1)
    c = np.minimum(((np.arange(width) + 0.5) * arr.shape[1] / width).astype(np.intp), arr.shape[1] - 1)
    return arr[r][:, c]


def _source_counts(state: _State, H: int, W: int) -> np.ndarray:
    flat = state.src_rows.astype(np.int64).ravel() * W + state.src_cols.ravel()
    return np.bincount(flat, minlength=H * W).reshape(H, W).astype(np.int32)


def resize(image: np.ndarray,
           target_width: Optional[int] = None,
           target_height: Optional[int] = None,
           *,
           energy: str = "forward",
           pixel_energy: Optional[np.ndarray] = None,
           energy_fn: Optional[EnergyFn] = None,
           guard_fn: Optional[GuardFn] = None,
           stop_fn: Optional[StopFn] = None,
           order: str = "greedy",
           finish: str = "scale",
           crop_importance: Optional[np.ndarray] = None,
           max_insert_fraction: float = 0.5) -> CarveResult:
    """
    Content-aware resize to (target_width, target_height); None keeps a dimension.

    Pipeline: (1) remove seams on every shrinking axis, consulting guard_fn
    before and stop_fn after each one; (2) insert seams on every growing axis;
    (3) if carving stopped early, finish with ``finish``: "scale" (INTER_AREA),
    "crop" (window maximising ``crop_importance``, an (H, W) map on the original;
    centre crop if None) or "none" (return the partially carved image).

    energy          "forward" (Rubinstein 2008) or "backward" (Avidan & Shamir 2007)
    pixel_energy    optional (H, W) static extra cost (finite, >= 0)
    energy_fn       optional f(CarveView) -> (h, w) extra cost, recomputed every seam
    guard_fn        optional f(CarveView, seam_mask) -> bool; False stops carving
    stop_fn         optional f(history) -> bool, called after every removed seam
    order           axis order when both dimensions shrink
    """
    t0 = time.perf_counter()
    img = np.asarray(image)
    if img.ndim not in (2, 3):
        raise ValueError(f"image must be (H, W) or (H, W, C), got shape {img.shape}")
    H, W = img.shape[:2]
    tw = W if target_width is None else int(target_width)
    th = H if target_height is None else int(target_height)
    if tw < 1 or th < 1:
        raise ValueError("target size must be at least 1x1")
    if energy not in ENERGIES:
        raise ValueError(f"energy must be one of {ENERGIES}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if finish not in FINISHES:
        raise ValueError(f"finish must be one of {FINISHES}")
    if not 0.0 < max_insert_fraction <= 1.0:
        raise ValueError("max_insert_fraction must be in (0, 1]")

    pix = None
    if pixel_energy is not None:
        pix = np.array(pixel_energy, dtype=np.float64)          # copy: never mutate caller's mask
        if pix.shape != (H, W):
            raise ValueError(f"pixel_energy must have shape {(H, W)}, got {pix.shape}")
        if not np.isfinite(pix).all() or (pix < 0).any():
            raise ValueError("pixel_energy must be finite and non-negative "
                             "(use a large finite weight, not inf, for hard protection)")
    if crop_importance is not None and np.shape(crop_importance) != (H, W):
        raise ValueError(f"crop_importance must have shape {(H, W)}")

    src_rows, src_cols = np.indices((H, W), dtype=np.int32)
    state = _State(img.copy(), to_gray(img), src_rows, src_cols, pix)

    history: list[SeamRecord] = []
    state, reason = _remove_seams(state, max(W - tw, 0), max(H - th, 0), order=order,
                                  energy=energy, energy_fn=energy_fn, guard_fn=guard_fn,
                                  stop_fn=stop_fn, history=history)
    removed = {VERTICAL: sum(r.axis == VERTICAL for r in history),
               HORIZONTAL: sum(r.axis == HORIZONTAL for r in history)}

    inserted = {VERTICAL: max(tw - W, 0), HORIZONTAL: max(th - H, 0)}
    for axis, k in inserted.items():
        if k:
            state = _insert_seams(state, k, axis, energy, max_insert_fraction)

    carved_shape = state.shape
    finish_applied = None
    if carved_shape != (th, tw) and finish == "crop":
        ch, cw = min(carved_shape[0], th), min(carved_shape[1], tw)
        imp = np.zeros(carved_shape) if crop_importance is None else \
            np.asarray(crop_importance, dtype=np.float64)[state.src_rows, state.src_cols]
        y0, x0 = best_crop_window(imp, ch, cw)
        state = state.crop(y0, x0, ch, cw)
        finish_applied = "crop"

    source_counts = _source_counts(state, H, W)

    out_img, out_rows, out_cols = state.image, state.src_rows, state.src_cols
    if state.shape != (th, tw) and finish in ("scale", "crop"):
        out_img = cv2.resize(out_img, (tw, th), interpolation=cv2.INTER_AREA)
        out_img = out_img.reshape((th, tw) + img.shape[2:])    # cv2 drops a singleton channel
        out_rows = _nearest_resample(out_rows, th, tw)
        out_cols = _nearest_resample(out_cols, th, tw)
        finish_applied = "scale" if finish_applied is None else finish_applied + "+scale"

    return CarveResult(out_img, out_rows, out_cols, source_counts, history, removed, inserted,
                       reason is not None, reason, finish_applied, carved_shape,
                       time.perf_counter() - t0, state.image, state.src_rows, state.src_cols)


# =========================================================================== #
# Visualisation helper
# =========================================================================== #

def draw_seam_overlay(original: np.ndarray, source_counts: np.ndarray,
                      removed_color=(0, 0, 255), inserted_color=(0, 255, 0)) -> np.ndarray:
    """Original image (BGR uint8) with removed pixels in red and duplicated pixels in green."""
    img = np.asarray(original)
    if not np.issubdtype(img.dtype, np.integer):
        img = np.clip(img * 255.0, 0, 255)
    elif img.dtype != np.uint8:
        img = img / (np.iinfo(img.dtype).max / 255.0)
    img = img.astype(np.uint8)
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img.reshape(img.shape[:2]), cv2.COLOR_GRAY2BGR)
    out = img[..., :3].copy()
    out[source_counts == 0] = removed_color
    out[source_counts >= 2] = inserted_color
    return out


# =========================================================================== #
# CLI
# =========================================================================== #

def _parse_size(text: Optional[str], current: int) -> int:
    if text is None:
        return current
    text = text.strip()
    value = round(current * float(text[:-1]) / 100.0) if text.endswith("%") else int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"size {text!r} gives less than 1 pixel")
    return value


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Plain seam-carving resize (forward or backward energy).")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--width", help="target width in pixels, or a percentage such as 75%%")
    ap.add_argument("--height", help="target height in pixels, or a percentage such as 75%%")
    ap.add_argument("--energy", choices=ENERGIES, default="forward")
    ap.add_argument("--order", choices=ORDERS, default="greedy")
    ap.add_argument("--overlay", help="also save the original with removed (red) / inserted (green) pixels")
    args = ap.parse_args(argv)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        ap.error(f"could not read {args.input}")
    H, W = img.shape[:2]
    res = resize(img, _parse_size(args.width, W), _parse_size(args.height, H),
                 energy=args.energy, order=args.order)
    if not cv2.imwrite(args.output, res.image):
        ap.error(f"could not write {args.output}")
    if args.overlay and not cv2.imwrite(args.overlay, draw_seam_overlay(img, res.source_counts)):
        ap.error(f"could not write {args.overlay}")
    oh, ow = res.image.shape[:2]
    print(f"{W}x{H} -> {ow}x{oh} | removed {res.seams_removed} | inserted {res.seams_inserted} "
          f"| {res.elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
