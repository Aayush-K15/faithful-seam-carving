import numpy as np
import pytest

from fsc import conditions
from fsc.engine import resize
from fsc.reverse import embed_png, extract_png, pack, psnr, restore, side_file_sizes


def _scene(gray=False, seed=0):
    rng = np.random.default_rng(seed)
    H, W = 36, 60
    yy, xx = np.mgrid[:H, :W]
    img = (40 + 0.8 * xx + 0.5 * yy + rng.normal(0, 3, (H, W)))[..., None] * np.array([1.0, 0.8, 0.6])
    lab = np.zeros((H, W), np.int32)
    for k, cx in enumerate((15, 42), start=1):
        m = (yy - 18) ** 2 + (xx - cx) ** 2 < 8 ** 2
        lab[m] = k
        img[m] = (200 - 40 * k) + rng.normal(0, 10, (int(m.sum()), 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return (img[..., 0].copy() if gray else img), lab


@pytest.mark.parametrize("method", ["faithful", "sc_forward", "static_mask", "scale", "crop"])
def test_exact_round_trip(method):
    img, lab = _scene()
    res = conditions.run(method, img, lab, 30)
    back = restore(res.image, pack(img, res, "exact"))
    assert back.exact and back.output_matches
    assert np.array_equal(back.image, img)


def test_round_trip_through_png_and_grayscale():
    img, lab = _scene(gray=True)
    res = conditions.run("faithful", img, lab, 30)
    png = embed_png(res.image, pack(img, res, "exact"))
    shown, data = extract_png(png)
    assert np.array_equal(shown, res.image)
    back = restore(shown, data)
    assert back.exact and np.array_equal(back.image, img)


def test_height_reduction_and_enlarging_round_trip():
    img, _ = _scene(seed=1)
    for tw, th in ((45, 28), (80, None)):
        res = resize(img, tw, th)
        back = restore(res.image, pack(img, res, "exact"))
        assert np.array_equal(back.image, img)


def test_small_and_tiny_modes():
    img, lab = _scene(seed=2)
    res = conditions.run("sc_forward", img, lab, 40)        # carving only, no scaling
    small = restore(res.image, pack(img, res, "small"))
    assert small.exact and np.array_equal(small.image, img)
    tiny = restore(res.image, pack(img, res, "tiny"))
    assert not tiny.exact
    assert tiny.guessed.sum() == 20 * img.shape[0]         # one guessed pixel per row per seam
    assert np.array_equal(tiny.image[~tiny.guessed], img[~tiny.guessed])
    assert psnr(tiny.image, img) > 25
    sizes = [side_file_sizes(pack(img, res, m))["total"] for m in ("exact", "small", "tiny")]
    assert sizes[2] < sizes[1] <= sizes[0]


def test_changed_image_is_detected():
    img, lab = _scene()
    res = conditions.run("faithful", img, lab, 30)
    data = pack(img, res, "exact")
    changed = res.image.copy()
    changed[0, 0] ^= 1
    back = restore(changed, data)
    assert not back.output_matches and not back.exact
    with pytest.raises(ValueError):
        restore(res.image[:, :-1], data)
