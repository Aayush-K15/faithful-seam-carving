"""
Download COCO128-seg (128 COCO images with instance polygons, ~7 MB, GPL-3.0,
from Ultralytics) into data/coco128-seg.

    python scripts/get_data.py

For a larger study, convert COCO val2017 to the same YOLO-seg layout
(images/<split>/*.jpg, labels/<split>/*.txt) and pass --data/--split to the scripts.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"


def main():
    dest = Path(__file__).resolve().parents[1] / "data"
    if (dest / "coco128-seg").exists():
        print(f"already present: {dest / 'coco128-seg'}")
        return
    dest.mkdir(exist_ok=True)
    print(f"downloading {URL}")
    blob = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for n in z.namelist():
            if n.startswith("/") or ".." in Path(n).parts:
                sys.exit(f"refusing suspicious path in archive: {n}")
        z.extractall(dest)
    print(f"done: {dest / 'coco128-seg'}")


if __name__ == "__main__":
    main()
