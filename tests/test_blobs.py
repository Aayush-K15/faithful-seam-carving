import numpy as np

from fsc.blobs import blob_labels


def test_blob_labels_finds_separate_discs_on_plain_background():
    img = np.full((120, 200), 30, np.uint8)
    yy, xx = np.mgrid[:120, :200]
    for cx in (40, 100, 160):
        img[(yy - 60) ** 2 + (xx - cx) ** 2 < 20 ** 2] = 200
    lab = blob_labels(img)
    assert lab.max() == 3
    assert lab[60, 40] != lab[60, 100] != lab[60, 160]
    assert lab[0, 0] == 0
