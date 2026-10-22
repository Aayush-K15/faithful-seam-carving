import numpy as np
import pytest

from fsc import baselines, conditions
from fsc.datasets import draw_label_map, load_yolo_seg
from fsc.engine import resize
from fsc.metrics import image_metrics, per_object


def scene():
    rng = np.random.default_rng(1)
    img = (rng.random((40, 60, 3)) * 60).astype(np.uint8)
    lab = np.zeros((40, 60), dtype=np.int32)
    lab[10:30, 10:20] = 1
    lab[10:30, 24:34] = 2
    lab[10:30, 45:55] = 3
    img[lab > 0] = 230
    return img, lab


def test_scaling_metrics():
    img, lab = scene()
    res = baselines.scale(img, 30)
    po = per_object(lab, res)
    assert po["kept"].min() == 1.0 and not po["destroyed"].any() and not po["wts"].any()
    assert np.allclose(po["aspect"], np.log(2), atol=0.15)
    row = image_metrics(lab, res, in_width=60, in_height=40)
    assert row["intact_rate"] == 1.0 and row["carved_fraction"] == 0.0


def test_crop_vanishes_and_damages():
    img, lab = scene()
    po = per_object(lab, baselines.crop(img, 30))      # centre crop: columns 15..44
    assert po["vanished"].sum() == 1                    # object 3 (cols 45..54) is cut out
    assert po["kept"][0] == pytest.approx(0.5)          # object 1 cut in half
    assert po["destroyed"][2]


def test_merge_detected():
    img, lab = scene()
    # hand-made result: delete columns 20..23 (the gap between objects 1 and 2)
    keep = np.r_[0:20, 24:60]
    r, c = np.indices((40, 56), dtype=np.int32)
    c = keep[c].astype(np.int32)
    counts = np.ones((40, 60), np.int32)
    counts[:, 20:24] = 0
    res = baselines.scale(img, 56)
    res.src_rows, res.src_cols, res.source_counts = r, c, counts
    res.image = img[r, c]
    po = per_object(lab, res)
    assert po["merged"][:2].all() and not po["merged"][2] and int(po["new_pairs"]) == 1


def test_seam_carving_metrics_exact():
    img, lab = scene()
    res = resize(img, 40)
    po = per_object(lab, res)
    kept_direct = [(res.source_counts[lab == i] > 0).mean() for i in (1, 2, 3)]
    assert np.allclose(po["kept"], kept_direct)


@pytest.mark.parametrize("name", conditions.ALL)
def test_every_condition_hits_target_size(name):
    img, lab = scene()
    res = conditions.run(name, img, lab, 36)
    assert res.image.shape == (40, 36, 3)
    row = image_metrics(lab, res, in_width=60, in_height=40)
    assert row["n_objects"] == 3


def test_draw_label_map_small_on_top_and_filter():
    big = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float)
    small = np.array([[5, 5], [10, 5], [10, 10], [5, 10]], float)
    tiny = np.array([[30, 30], [31, 30], [31, 31]], float)
    lab, cls = draw_label_map((40, 40), [small, big, tiny], [7, 3, 9], min_area_px=4)
    assert lab.max() == 2 and cls == [3, 7]
    assert lab[7, 7] == 2 and lab[15, 15] == 1          # the small one stays visible


def test_load_yolo_seg(tmp_path):
    import cv2
    img = np.zeros((50, 100, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    (tmp_path / "a.txt").write_text("0 0.1 0.1 0.4 0.1 0.4 0.9 0.1 0.9\n2 0.6 0.2 0.9 0.2 0.9 0.8\n")
    s = load_yolo_seg(tmp_path / "a.jpg", tmp_path / "a.txt", min_area_frac=0.0)
    assert s.n_objects == 2 and sorted(s.classes) == [0, 2]
    assert s.labels[25, 25] > 0 and s.labels[45, 5] == 0


def test_object_detail_identity_scaling_and_carving():
    from fsc.metrics import object_detail
    img = np.full((60, 80), 40, np.uint8)
    lab = np.zeros((60, 80), np.int32)
    tex = (np.add.outer(np.sin(np.arange(30) / 3.0), np.cos(np.arange(30) / 2.0)) * 60 + 128).astype(np.uint8)
    img[15:45, 25:55] = tex
    lab[15:45, 25:55] = 1
    same = resize(img, 80)
    assert object_detail(img, lab, same)[0] == pytest.approx(1.0)
    squashed = baselines.scale(img, 60)
    # seams forced through the object: make the object the only cheap region
    from fsc.engine import resize as carve
    cut = carve(img, 60, pixel_energy=np.where(lab > 0, 0.0, 100.0))
    d_scale = object_detail(img, lab, squashed)[0]
    d_cut = object_detail(img, lab, cut)[0]
    assert d_scale > 0.8 and d_cut < d_scale
