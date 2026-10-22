import numpy as np
import pytest

from fsc import engine as ce
from fsc.integrity import (InstanceIntegrity, contact_pairs, gap_mask, object_mask,
                           remove_masked)


def two_objects(gap=2):
    """Two 6-px-wide objects separated by `gap` background columns, on a 20x30 map."""
    lab = np.zeros((20, 30), dtype=np.int32)
    lab[4:16, 5:11] = 1
    lab[4:16, 11 + gap:17 + gap] = 2
    return lab


def textured(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(shape + (3,)) * 255).astype(np.uint8)


def test_contact_pairs():
    lab = two_objects(gap=0)
    assert contact_pairs(lab) == {(1, 2)}
    assert contact_pairs(two_objects(gap=1)) == set()


def test_gap_mask_rows_and_columns():
    lab = two_objects(gap=2)
    g = gap_mask(lab, 3, ce.VERTICAL)
    assert g[4:16, 11:13].all() and g.sum() == 12 * 2
    assert not gap_mask(lab, 1, ce.VERTICAL).any()          # gap of 2 > 1
    assert not gap_mask(lab, 3, ce.HORIZONTAL).any()        # no vertical gaps here
    np.testing.assert_array_equal(gap_mask(lab.T, 3, ce.HORIZONTAL), g.T)


def test_remove_masked_matches_seam_removal():
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 9, (7, 9))
    seam = np.array([0, 1, 2, 2, 3, 4, 4])
    mask = ce.seam_mask(seam, ce.VERTICAL, arr.shape)
    np.testing.assert_array_equal(remove_masked(arr, mask, ce.VERTICAL), ce.remove_vertical_seam(arr, seam))
    mask_h = ce.seam_mask(np.array([0, 1, 1, 2, 3, 3, 4, 5, 6]), ce.HORIZONTAL, arr.shape)
    assert remove_masked(arr, mask_h, ce.HORIZONTAL).shape == (6, 9)


def view_of(shape, step=0, axis=ce.VERTICAL):
    r, c = np.indices(shape, dtype=np.int32)
    return ce.CarveView(np.zeros(shape), r, c, axis, step)


def test_budget_cost_grows_with_deletion():
    lab = two_objects()
    ctrl = InstanceIntegrity(lab, lam=1.0, gamma=8.0, gap_cost=0.0)
    e0 = ctrl.energy(view_of(lab.shape))
    assert e0[lab == 1].min() == pytest.approx(1.0) and e0[lab == 0].max() == 0.0
    # pretend half of object 1 is gone: view whose sources skip 3 of its 6 columns
    r, c = np.indices((20, 27), dtype=np.int32)
    c = np.where(c >= 5, c + 3, c)                          # columns 5..7 of the original removed
    view = ce.CarveView(np.zeros((20, 27)), r, c, ce.VERTICAL, 1)
    e1 = ctrl.energy(view)
    lab_now = view.gather(lab)
    assert e1[lab_now == 1].min() == pytest.approx(np.exp(8 * 0.5))
    assert e1[lab_now == 2].min() == pytest.approx(1.0)


def test_guard_refuses_merge_and_overdeletion():
    lab = two_objects(gap=1)
    ctrl = InstanceIntegrity(lab, max_loss=0.5)
    v = view_of(lab.shape)
    through_gap = ce.seam_mask(np.full(20, 11), ce.VERTICAL, lab.shape)
    assert not ctrl.guard(v, through_gap) and ctrl.refusals["merge"] == 1
    through_bg = ce.seam_mask(np.full(20, 25), ce.VERTICAL, lab.shape)
    assert ctrl.guard(v, through_bg)
    # object 1 already lost 3 of 6 columns: one more column exceeds 50%
    r, c = np.indices((20, 27), dtype=np.int32)
    c = np.where(c >= 5, c + 3, c)
    v2 = ce.CarveView(np.zeros((20, 27)), r, c, ce.VERTICAL, 3)
    col_of_obj1 = int(np.flatnonzero(v2.gather(lab)[10] == 1)[0])
    assert not ctrl.guard(v2, ce.seam_mask(np.full(20, col_of_obj1), ce.VERTICAL, (20, 27)))
    assert ctrl.refusals["loss"] == 1


def test_faithful_resize_never_destroys_or_fuses():
    lab = np.zeros((40, 60), dtype=np.int32)
    for k, x in enumerate(range(4, 56, 10), start=1):     # 6 objects, 6 px wide, 4 px apart
        lab[8:32, x:x + 6] = k
    img = textured((40, 60))
    img[lab > 0] = 255                                      # bright objects: costly to cross
    ctrl = InstanceIntegrity(lab)
    res = ce.resize(img, 24, energy_fn=ctrl.energy, guard_fn=ctrl.guard)
    out_lab = lab[res.src_rows, res.src_cols]
    assert contact_pairs(out_lab) == set()
    kept = np.bincount(lab.ravel(), weights=(res.source_counts > 0).ravel(), minlength=7)[1:]
    area = np.bincount(lab.ravel(), minlength=7)[1:]
    assert (kept / area >= 0.5).all()
    assert res.image.shape == (40, 24, 3)


def test_object_mask():
    assert object_mask(two_objects()).sum() == 2 * 12 * 6


def test_rejects_bad_labels():
    with pytest.raises(ValueError):
        InstanceIntegrity(-np.ones((4, 4), dtype=int))


def test_object_removal_deletes_target_and_spares_neighbour():
    from fsc.removal import remove_objects
    lab = np.zeros((30, 60), dtype=np.int32)
    lab[5:25, 10:20] = 1
    lab[5:25, 24:30] = 2          # target, right next to object 1
    lab[5:25, 40:50] = 3
    img = textured((30, 60), seed=4)
    img[lab == 1] = 250
    img[lab == 3] = 250
    r = remove_objects(img, lab, [2])
    assert r.complete and r.image.shape == img.shape
    out_lab = lab[r.carved.src_rows, r.carved.src_cols]
    assert not (out_lab == 2).any()
    for k in (1, 3):
        assert (out_lab == k).sum() / (lab == k).sum() >= 0.9
