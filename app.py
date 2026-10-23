"""
Web front end for Faithful Seam Carving.

    pip install -r requirements.txt streamlit
    pip install ultralytics        # optional: automatic object masks
    streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from fsc import conditions
from fsc.metrics import image_metrics
from fsc.viz import object_overlay

st.set_page_config(page_title="Faithful Seam Carving", layout="wide")
st.title("Faithful Seam Carving")
st.caption("Resize a photo without crushing or fusing the objects in it.")

METHODS = {
    "Faithful (ours)": "faithful",
    "Plain seam carving (forward energy)": "sc_forward",
    "Plain seam carving (original)": "sc_backward",
    "Uniform scaling": "scale",
    "Centre crop": "crop",
}

up = st.sidebar.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
width_pct = st.sidebar.slider("New width (% of original)", 30, 100, 50, 5)
method_names = st.sidebar.multiselect("Methods to compare", list(METHODS), default=list(METHODS)[:2])
max_side = st.sidebar.select_slider("Work at max size (px)", [240, 320, 480, 640], value=320,
                                    help="Smaller is faster. Seam carving is slow in pure Python.")
use_yolo = st.sidebar.checkbox("Detect objects with YOLO", value=True)
show_obj = st.sidebar.checkbox("Show detected objects", value=True)
go = st.sidebar.button("Run", type="primary", disabled=up is None)

if up is None:
    st.info("Upload a photo in the sidebar to start.")
    st.stop()

img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read that image.")
    st.stop()
s = max_side / max(img.shape[:2])
if s < 1:
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
H, W = img.shape[:2]
tw = max(8, round(W * width_pct / 100))


@st.cache_data(show_spinner="Detecting objects...")
def detect(data: bytes, size: int):
    from fsc.datasets import predict_instances
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    k = size / max(im.shape[:2])
    if k < 1:
        im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    labels, _ = predict_instances(im)
    return labels


labels = np.zeros((H, W), np.int32)
if use_yolo:
    try:
        labels = detect(up.getvalue(), max_side)
        st.sidebar.success(f"{int(labels.max())} objects detected")
    except ImportError:
        st.sidebar.warning("YOLO not installed (pip install ultralytics). "
                           "Running without object masks, so Faithful cannot protect anything.")

if use_yolo and labels.max() == 0:
    st.warning("No objects detected. YOLO only knows the 80 COCO classes (people, animals, cups, cars...). "
               "With nothing to protect, Faithful behaves like plain seam carving, so the results will look the same. "
               "Try a photo with several people, animals or everyday objects.")

c0, c1 = st.columns(2)
c0.subheader(f"Original ({W}x{H})")
c0.image(cv2.cvtColor(object_overlay(img, labels) if show_obj and labels.max() else img,
                      cv2.COLOR_BGR2RGB))
if not go:
    c1.info("Press Run.")
    st.stop()

cols = st.columns(max(1, len(method_names)))
for col, name in zip(cols, method_names):
    key = METHODS[name]
    with st.spinner(f"{name}..."):
        res = conditions.run(key, img, labels, tw)
    col.subheader(name)
    out = res.image
    col.image(cv2.cvtColor(object_overlay(out, labels[res.src_rows, res.src_cols])
                           if show_obj and labels.max() else out, cv2.COLOR_BGR2RGB))
    if labels.max():
        m = image_metrics(labels, res, in_width=W, in_height=H, image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        col.caption(f"{out.shape[1]}x{out.shape[0]} | crushed {m['destroyed']} | fused {m['merged_objects']} | "
                    f"detail {m['detail']:.2f} | {m['seconds']:.1f}s")
    ok, buf = cv2.imencode(".png", out)
    col.download_button("Download", buf.tobytes(), f"{key}_{width_pct}.png", "image/png", key=f"dl_{key}")
