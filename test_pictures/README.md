# Test pictures

Sample photos for trying the app (`streamlit run app.py`) or `scripts/demo.py`.
They come from scikit-image's bundled sample data (`skimage.data`); each image's
source and licence are listed in the scikit-image documentation:
https://scikit-image.org/docs/stable/api/skimage.data.html

| File | Good for |
|---|---|
| `coins.jpg` | many separate objects: plain seam carving crushes coins, ours keeps all 24 |
| `coffee.jpg` | a few objects on a table |
| `rocket.jpg` | one large object, wide scene |
| `astronaut.jpg`, `chelsea.jpg` | people and animals; few objects, so little difference between methods |

Object masks need `pip install ultralytics` (YOLO). Try 50% width.
