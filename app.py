"""
Web front end for Faithful Seam Carving.

    pip install -r requirements.txt
    pip install ultralytics        # optional: automatic object masks
    streamlit run app.py

Tab "Resize": upload a photo, compare methods, download the result as a plain
PNG or as a reversible PNG that carries what is needed to undo the resize.
Tab "Restore": upload a reversible PNG and get the original back.
"""
import hashlib

import cv2
import numpy as np
import streamlit as st

from fsc import conditions
from fsc.metrics import image_metrics
from fsc.reverse import embed_png, extract_png, pack, psnr, restore, side_file_sizes
from fsc.viz import object_overlay

st.set_page_config(page_title="Faithful Seam Carving", layout="wide")
st.title("Faithful Seam Carving")
st.caption("Resize a photo without crushing or fusing the objects in it, and undo the resize later.")

METHODS = {
    "Faithful (ours)": "faithful",
    "Plain seam carving (forward energy)": "sc_forward",
    "Plain seam carving (original)": "sc_backward",
    "Uniform scaling": "scale",
    "Centre crop": "crop",
}
REVERSIBLE = {
    "Exact: bit-for-bit original, biggest file": "exact",
    "Small: removed pixels exact, scaled part approximate": "small",
    "Tiny: a few KB, everything removed is guessed": "tiny",
}


def png_bytes(img):
    return cv2.imencode(".png", img)[1].tobytes()


def kb(n):
    return f"{n / 1024:.1f} KB"


@st.cache_data(show_spinner="Detecting objects...")
def detect(data: bytes, size: int):
    from fsc.datasets import predict_instances
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    k = size / max(im.shape[:2])
    if k < 1:
        im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    labels, _ = predict_instances(im)
    return labels


def resize_tab():
    with st.sidebar:
        st.header("Resize settings")
        up = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
        width_pct = st.slider("New width (% of original)", 30, 100, 50, 5)
        method_names = st.multiselect("Methods to compare", list(METHODS), default=list(METHODS)[:2])
        max_side = st.select_slider("Work at max size (px)", [240, 320, 480, 640], value=320,
                                    help="Smaller is faster. Seam carving is slow in pure Python.")
        mask_mode = st.radio("Find objects with", ["YOLO (people, animals, everyday objects)",
                                                   "Simple blobs (coins, parts on a plain background)"],
                             help="YOLO knows 80 COCO classes only. Blobs thresholds the image and splits "
                                  "touching shapes.")
        show_obj = st.checkbox("Show detected objects", value=True)
        rev_label = st.selectbox("Reversible download", list(REVERSIBLE),
                                 help="What the reversible PNG carries so the resize can be undone.")
        go = st.button("Run", type="primary", disabled=up is None)

    if up is None:
        st.info("Upload a photo in the sidebar to start.")
        return

    img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read that image.")
        return
    s = max_side / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]
    tw = max(8, round(W * width_pct / 100))

    labels = np.zeros((H, W), np.int32)
    if mask_mode.startswith("Simple"):
        from fsc.blobs import blob_labels
        labels = blob_labels(img)
        st.sidebar.success(f"{int(labels.max())} blobs found")
    else:
        try:
            labels = detect(up.getvalue(), max_side)
            st.sidebar.success(f"{int(labels.max())} objects detected")
        except ImportError:
            st.sidebar.warning("YOLO not installed (pip install ultralytics). "
                               "Running without object masks, so Faithful cannot protect anything.")
    if labels.max() == 0:
        st.warning("No objects found, so Faithful has nothing to protect and behaves like plain seam carving. "
                   "YOLO only knows 80 COCO classes (people, animals, cups, cars...). For coins or parts on a "
                   "plain background switch to 'Simple blobs' in the sidebar.")

    c0, c1 = st.columns(2)
    c0.subheader(f"Original ({W}x{H})")
    c0.image(cv2.cvtColor(object_overlay(img, labels) if show_obj and labels.max() else img, cv2.COLOR_BGR2RGB))
    c0.caption(f"PNG of this image: {kb(len(png_bytes(img)))}")

    # keep results across reruns (a download click reruns the script)
    key = hashlib.sha1(up.getvalue() + repr((width_pct, max_side, mask_mode, tuple(method_names))).encode()).hexdigest()
    if go:
        results = {}
        for name in method_names:
            with st.spinner(f"{name}..."):
                results[name] = conditions.run(METHODS[name], img, labels, tw)
        st.session_state["results"] = (key, results)
    stored = st.session_state.get("results")
    if not stored or stored[0] != key:
        c1.info("Press Run.")
        return
    results = stored[1]
    mode = REVERSIBLE[rev_label]

    cols = st.columns(max(1, len(results)))
    for col, (name, res) in zip(cols, results.items()):
        m_key = METHODS[name]
        out = res.image
        col.subheader(name)
        col.image(cv2.cvtColor(object_overlay(out, labels[res.src_rows, res.src_cols])
                               if show_obj and labels.max() else out, cv2.COLOR_BGR2RGB))
        if labels.max():
            m = image_metrics(labels, res, in_width=W, in_height=H, image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            col.caption(f"{out.shape[1]}x{out.shape[0]} | crushed {m['destroyed']} | fused {m['merged_objects']} | "
                        f"detail {m['detail']:.2f} | {m['seconds']:.1f}s")
        plain = png_bytes(out)
        col.download_button(f"Download PNG ({kb(len(plain))})", plain, f"{m_key}_{width_pct}.png", "image/png",
                            key=f"dl_{m_key}")
        side = pack(img, res, mode)
        check = restore(out, side)
        col.download_button(f"Download reversible PNG ({kb(len(plain) + len(side))})", embed_png(out, side),
                            f"{m_key}_{width_pct}_reversible.png", "image/png", key=f"rev_{m_key}")
        q = "bit-exact" if check.exact else f"approximate, PSNR {psnr(check.image, img):.1f} dB"
        col.caption(f"Undo data: +{kb(side_file_sizes(side)['total'])} ({mode}); restores {q}.")


def restore_tab():
    st.write("Upload a reversible PNG made in the Resize tab to get the original back. "
             "Send the PNG as a file: chat apps that re-compress or shrink photos break it.")
    up = st.file_uploader("Reversible PNG", type=["png"], key="restore_upload")
    if up is None:
        return
    try:
        shown, side = extract_png(up.getvalue())
    except ValueError as e:
        st.error(str(e))
        return
    if side is None:
        st.error("This PNG carries no undo data. Download it with 'Download reversible PNG'.")
        return
    try:
        back = restore(shown, side)
    except ValueError as e:
        st.error(str(e))
        return
    if not back.output_matches:
        st.warning("The resized image was changed after it was made (re-saved or edited), so the result is "
                   "approximate.")
    c0, c1 = st.columns(2)
    c0.subheader(f"Received ({shown.shape[1]}x{shown.shape[0]})")
    c0.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB) if shown.ndim == 3 else shown)
    c1.subheader(f"Restored ({back.image.shape[1]}x{back.image.shape[0]})")
    c1.image(cv2.cvtColor(back.image, cv2.COLOR_BGR2RGB) if back.image.ndim == 3 else back.image)
    how = "bit-for-bit identical to the original" if back.exact else \
        f"approximate ({int(back.guessed.sum())} pixels guessed)" if back.guessed.any() else \
        "approximate (the scaled part is blurred)"
    c1.caption(f"Mode: {back.mode} | {how} | undo data {kb(len(side))}")
    c1.download_button("Download restored PNG", png_bytes(back.image), "restored.png", "image/png")


tab_resize, tab_restore = st.tabs(["Resize", "Restore"])
with tab_resize:
    resize_tab()
with tab_restore:
    restore_tab()
