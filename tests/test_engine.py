"""
Tests for fsc.engine.   Run from the project root:  python -m pytest -q

The key checks compare the vectorised engine with independent references:
  * the forward-energy terms against a per-pixel transcription of the equations;
  * the DP minimum against brute-force enumeration of every seam;
  * the forward-energy cost of a seam against what it actually means: the sum of
    intensity jumps between pixels that become neighbours only after removal.
"""

import itertools

import numpy as np
import pytest

from fsc import engine as ce


# --------------------------------------------------------------------------- #
# Independent references
# --------------------------------------------------------------------------- #

def naive_forward_terms(gray):
    H, W = gray.shape

    def I(i, j):  # clamped indexing, as documented in forward_energy_terms
        return gray[min(max(i, 0), H - 1), min(max(j, 0), W - 1)]

    c_u, c_l, c_r = (np.zeros((H, W)) for _ in range(3))
    for i in range(H):
        for j in range(W):
            h = abs(I(i, j + 1) - I(i, j - 1))
            c_u[i, j] = h
            c_l[i, j] = h + abs(I(i - 1, j) - I(i, j - 1))
            c_r[i, j] = h + abs(I(i - 1, j) - I(i, j + 1))
    return c_u, c_l, c_r


def all_vertical_seams(H, W, lo=0, hi=None):
    hi = W - 1 if hi is None else hi
    for start in range(lo, hi + 1):
        for moves in itertools.product((-1, 0, 1), repeat=H - 1):
            seam = [start]
            for m in moves:
                seam.append(seam[-1] + m)
            if all(lo <= s <= hi for s in seam):
                yield np.array(seam)


def seam_cost_by_definition(terms, pix, seam):
    c_u, c_l, c_r = terms
    cost = c_u[0, seam[0]] + pix[0, seam[0]]
    for i in range(1, len(seam)):
        t = {-1: c_l, 0: c_u, 1: c_r}[seam[i - 1] - seam[i]]
        cost += t[i, seam[i]] + pix[i, seam[i]]
    return cost


def new_neighbour_cost(gray, seam):
    """Sum of |jumps| between pixels that are adjacent after removal but were not before."""
    H, W = gray.shape
    out = ce.remove_vertical_seam(gray, seam)
    cols = ce.remove_vertical_seam(np.tile(np.arange(W), (H, 1)), seam)
    cost = 0.0
    for i in range(H):
        for c in range(W - 2):
            if cols[i, c + 1] != cols[i, c] + 1:
                cost += abs(out[i, c + 1] - out[i, c])
    for i in range(H - 1):
        for c in range(W - 1):
            if cols[i, c] != cols[i + 1, c]:
                cost += abs(out[i + 1, c] - out[i, c])
    return cost


# --------------------------------------------------------------------------- #
# Energy and DP
# --------------------------------------------------------------------------- #

def test_forward_terms_match_equations():
    gray = np.random.default_rng(0).random((7, 9))
    for fast, slow in zip(ce.forward_energy_terms(gray), naive_forward_terms(gray)):
        np.testing.assert_allclose(fast, slow, atol=1e-12)


@pytest.mark.parametrize("energy", ce.ENERGIES)
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("with_pixel_energy", [False, True])
def test_dp_matches_brute_force(energy, seed, with_pixel_energy):
    rng = np.random.default_rng(seed)
    gray = rng.random((5, 6))
    pix = rng.random((5, 6)) * 2 if with_pixel_energy else np.zeros((5, 6))
    terms = ce.energy_terms(gray, energy)
    brute = min(seam_cost_by_definition(terms, pix, s) for s in all_vertical_seams(5, 6))

    seam = ce.find_vertical_seam(gray, pix if with_pixel_energy else None, energy)
    assert seam.cost == pytest.approx(brute, abs=1e-12)
    assert seam_cost_by_definition(terms, pix, seam.indices) == pytest.approx(seam.cost, abs=1e-12)
    assert seam.energy_cost + seam.pixel_cost == pytest.approx(seam.cost, abs=1e-12)
    assert np.all(np.abs(np.diff(seam.indices)) <= 1)


def test_forward_energy_is_the_energy_inserted_by_removal():
    """For interior seams the DP cost equals the new-neighbour jumps actually created."""
    gray = np.random.default_rng(1).random((5, 7))
    terms = ce.forward_energy_terms(gray)
    zero = np.zeros_like(gray)
    for seam in all_vertical_seams(5, 7, lo=1, hi=5):
        assert seam_cost_by_definition(terms, zero, seam) == pytest.approx(
            new_neighbour_cost(gray, seam), abs=1e-12)


def test_border_column_is_not_free():
    """Guards the documented clamping choice: carving must not degenerate into cropping."""
    gray = np.tile(np.linspace(0, 1, 20), (10, 1))       # pure horizontal ramp
    c_u, _, _ = ce.forward_energy_terms(gray)
    assert c_u[:, 0].min() > 0 and c_u[:, -1].min() > 0


# --------------------------------------------------------------------------- #
# Removal, both axes
# --------------------------------------------------------------------------- #

@pytest.fixture
def color_image():
    rng = np.random.default_rng(7)
    img = (rng.random((24, 31, 3)) * 255).astype(np.uint8)
    img[:, 10:14] = 255                                  # some structure, not only noise
    return img


@pytest.mark.parametrize("order", ce.ORDERS)
def test_dual_axis_reduction_shapes_and_source_maps(color_image, order):
    res = ce.resize(color_image, 20, 17, order=order)
    assert res.image.shape == (17, 20, 3)
    assert res.seams_removed == {ce.VERTICAL: 11, ce.HORIZONTAL: 7}
    # Pure removal: every output pixel is an untouched original pixel.
    np.testing.assert_array_equal(res.image, color_image[res.src_rows, res.src_cols])
    assert res.source_counts.sum() == 17 * 20
    assert set(np.unique(res.source_counts)) <= {0, 1}
    assert [r.step for r in res.history] == list(range(18))


def test_horizontal_is_transposed_vertical(color_image):
    a = ce.resize(color_image, target_height=15).image
    b = ce.resize(color_image.transpose(1, 0, 2), target_width=15).image.transpose(1, 0, 2)
    np.testing.assert_array_equal(a, b)


def test_grayscale_and_float_inputs(color_image):
    g = color_image[..., 1]
    assert ce.resize(g, 25).image.shape == (24, 25)
    f = color_image.astype(np.float32) / 255
    out = ce.resize(f, 25).image
    assert out.dtype == np.float32 and out.shape == (24, 25, 3)


def test_identity_when_size_unchanged(color_image):
    res = ce.resize(color_image, 31, 24)
    np.testing.assert_array_equal(res.image, color_image)
    assert res.history == []


def test_backward_energy_runs(color_image):
    assert ce.resize(color_image, 20, energy="backward").image.shape == (24, 20, 3)


# --------------------------------------------------------------------------- #
# Hooks: pixel_energy and stop_fn
# --------------------------------------------------------------------------- #

def test_pixel_energy_protects_region(color_image):
    pix = np.zeros(color_image.shape[:2])
    pix[:, 3:9] = 1e3
    res = ce.resize(color_image, 15, pixel_energy=pix)
    assert res.source_counts[:, 3:9].min() == 1          # nothing in the band was removed
    assert all(r.pixel_cost == 0 for r in res.history)


def test_pixel_energy_rejects_inf(color_image):
    pix = np.zeros(color_image.shape[:2])
    pix[0, 0] = np.inf
    with pytest.raises(ValueError):
        ce.resize(color_image, 20, pixel_energy=pix)


def test_stop_fn_then_scale(color_image):
    res = ce.resize(color_image, 15, stop_fn=lambda h: len(h) >= 3)
    assert res.stopped_early and len(res.history) == 3
    assert res.carved_shape == (24, 28)
    assert res.image.shape == (24, 15, 3)                # finished by scaling
    assert res.src_rows.shape == (24, 15)
    assert (res.source_counts == 0).sum() == 3 * 24      # exact: only the 3 seams removed


def test_stop_fn_without_finish(color_image):
    res = ce.resize(color_image, 15, stop_fn=lambda h: len(h) >= 3, finish="none")
    assert res.image.shape == (24, 28, 3)


def test_stop_on_last_seam_is_not_early(color_image):
    res = ce.resize(color_image, 28, stop_fn=lambda h: len(h) >= 3)
    assert not res.stopped_early and res.image.shape == (24, 28, 3)


# --------------------------------------------------------------------------- #
# Insertion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("extra", [1, 9, 40])                 # 40 > 50% -> several passes
def test_insertion_keeps_every_original_pixel(color_image, extra):
    H, W = color_image.shape[:2]
    res = ce.resize(color_image, W + extra)
    assert res.image.shape == (H, W + extra, 3)
    assert res.seams_inserted[ce.VERTICAL] == extra
    if extra <= W // 2:                                       # single pass: exact structure checks
        inserted = np.zeros(res.src_cols.shape, dtype=bool)
        inserted[:, 1:] = res.src_cols[:, 1:] == res.src_cols[:, :-1]
        assert (inserted.sum(axis=1) == extra).all()          # k disjoint seams per row
        recovered = res.image[~inserted].reshape(H, W, 3)
        np.testing.assert_array_equal(recovered, color_image)
        assert ((res.source_counts == 2).sum(axis=1) == extra).all()


def test_inserted_pixel_is_midpoint():
    img = np.array([[[0], [100], [200]]] * 4, dtype=np.uint8)   # (4, 3, 1)
    res = ce.resize(img, 4)
    for row, cols in zip(res.image[..., 0], res.src_cols):
        dup = int(np.flatnonzero(cols[1:] == cols[:-1])[0]) + 1
        left = row[dup - 1].astype(int)
        right = row[dup + 1].astype(int) if dup + 1 < len(row) else left
        assert row[dup] == np.rint((left + right) / 2)


def test_mixed_shrink_and_grow(color_image):
    res = ce.resize(color_image, 40, 18)
    assert res.image.shape == (18, 40, 3)
    assert res.seams_removed[ce.HORIZONTAL] == 6 and res.seams_inserted[ce.VERTICAL] == 9


# --------------------------------------------------------------------------- #
# Dynamic hooks: energy_fn, guard_fn, crop finish
# --------------------------------------------------------------------------- #

def test_energy_fn_called_every_seam_with_current_shape(color_image):
    seen = []

    def fn(view):
        seen.append((view.step, view.axis, view.shape))
        return np.zeros(view.shape)

    res = ce.resize(color_image, 27, 21, energy_fn=fn, order="width_first")
    removed = [(s, a) for s, a, _ in seen]
    assert len(set(removed)) == 4 + 3                     # one call per removed seam
    for step, axis, shape in seen:
        n_v = sum(1 for r in res.history[:step] if r.axis == ce.VERTICAL)
        n_h = step - n_v
        assert shape == (24 - n_h, 31 - n_v)


def test_energy_fn_zero_matches_no_hook(color_image):
    a = ce.resize(color_image, 22)
    b = ce.resize(color_image, 22, energy_fn=lambda v: np.zeros(v.shape))
    np.testing.assert_array_equal(a.image, b.image)


def test_energy_fn_bad_shape_rejected(color_image):
    with pytest.raises(ValueError):
        ce.resize(color_image, 22, energy_fn=lambda v: np.zeros((3, 3)))


def test_view_gather_maps_original_arrays(color_image):
    ids = np.arange(24 * 31).reshape(24, 31)

    def fn(view):
        g = view.gather(ids)
        np.testing.assert_array_equal(g, view.src_rows * 31 + view.src_cols)
        return None

    ce.resize(color_image, 25, 20, energy_fn=fn)


def test_guard_stops_and_scale_finishes(color_image):
    calls = []

    def guard(view, mask):
        assert mask.sum() == view.shape[0]                 # one pixel per row (vertical seam)
        calls.append(view.step)
        return view.step < 4

    res = ce.resize(color_image, 15, guard_fn=guard)
    assert res.stopped_early and res.stop_reason == "guard"
    assert len(res.history) == 4 and res.carved_shape == (24, 27)
    assert res.image.shape == (24, 15, 3) and res.finish_applied == "scale"


def test_crop_finish_is_exact_and_keeps_important_side(color_image):
    imp = np.zeros((24, 31))
    imp[:, :8] = 1.0                                       # importance on the left edge
    res = ce.resize(color_image, 15, guard_fn=lambda v, m: v.step < 3,
                    finish="crop", crop_importance=imp)
    assert res.image.shape == (24, 15, 3) and res.finish_applied == "crop"
    np.testing.assert_array_equal(res.image, color_image[res.src_rows, res.src_cols])
    assert res.source_counts[:, :8].min() == 1             # the important columns survive
    assert res.source_counts.sum() == 24 * 15


def test_best_crop_window_centres_ties():
    assert ce.best_crop_window(np.zeros((10, 20)), 10, 10) == (0, 5)
