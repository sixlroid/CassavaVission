"""
CassavaVision - cassava leaf instance segmentation (YOLO26-seg) web app.
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="CassavaVision | Cassava Leaf Segmentation",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants & Custom Class Display Mapping (Green, Brown, Yellow)
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"

DEFAULT_IMGSZ = 640
MAX_FILE_MB = 10
MAX_FILES = 6              # Maximum upload limit set to 6 images
MIN_SIDE = 64
MAX_SIDE = 1600
MIN_CONF = 0.05
MAX_DET = 100

CLASS_INFO = {
    "HealthyCassava": dict(
        label="Healthy", short="Healthy", color=(46, 184, 92), diseased=False, # Green
        desc="Uniform green leaf with no mosaic or streak pattern.",
    ),
    "MosaicDisease": dict(
        label="Mosaic Disease", short="CMD", color=(222, 184, 35), diseased=True, # Yellow
        desc="Yellow-green mosaic patches, often with curled or distorted leaves.",
    ),
    "BrownStreak": dict(
        label="Brown Streak Disease", short="CBSD", color=(139, 69, 19), diseased=True, # Brown
        desc="Yellow, feathery patches that follow the veins; browning in later stages.",
    ),
}
_FALLBACK_COLORS = [(46, 184, 92), (222, 184, 35), (139, 69, 19)]

def class_info(name: str) -> dict:
    if name in CLASS_INFO:
        return CLASS_INFO[name]
    return dict(
        label=name, short=name,
        color=_FALLBACK_COLORS[zlib.crc32(name.encode()) % len(_FALLBACK_COLORS)],
        diseased="healthy" not in name.lower(), desc="",
    )

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def show_image(img, caption: str | None = None) -> None:
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except Exception:
            continue

def find_model_path() -> Path | None:
    for p in (APP_DIR / "best.pt", APP_DIR / "weights" / "best.pt"):
        if p.is_file():
            return p
    if RUNS_DIR.is_dir():
        hits = [p for p in RUNS_DIR.rglob("best.pt") if p.is_file()]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None

@st.cache_resource(show_spinner="Loading segmentation model...")
def load_model(path_str: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(path_str)
    ckpt = getattr(model, "ckpt", None) or {}
    args = dict(ckpt.get("train_args") or {})
    try:
        params = int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        params = 0
    run_name = str(args.get("name") or "")
    arch = run_name.split("_")[0].replace("yolo", "YOLO", 1) if run_name else "YOLO26-seg"
    return dict(
        model=model,
        names={int(k): str(v) for k, v in dict(model.names).items()},
        args=args,
        arch=arch or "YOLO26-seg",
        imgsz=int(args.get("imgsz") or DEFAULT_IMGSZ),
        metrics=dict(ckpt.get("train_metrics") or {}),
        history=dict(ckpt.get("train_results") or {}),
        best_epoch=(int(ckpt["epoch"]) + 1) if isinstance(ckpt.get("epoch"), (int, float)) else None,
        params=params,
        size_mb=Path(path_str).stat().st_size / 1e6,
    )

def load_image(data: bytes) -> tuple[Image.Image | None, str | None, str | None]:
    if not data:
        return None, "The file is empty.", None
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"The file is {len(data) / 1e6:.1f} MB. Please upload an image under {MAX_FILE_MB} MB.", None
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or "").upper()
        im.load()
    except Exception:
        return None, "This file could not be read as an image. Please upload a JPG or PNG photo.", None
    if fmt not in {"JPEG", "PNG"}:
        return None, f"Unsupported image format ({fmt or 'unknown'}). Please upload a JPG or PNG.", None

    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    im = im.convert("RGB")

    w, h = im.size
    if min(w, h) < MIN_SIDE:
        return None, f"Image is too small ({w}x{h} px). Must be at least {MIN_SIDE}px.", None
    note = None
    if max(w, h) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        note = f"Large image was resized to {im.size[0]}x{im.size[1]} for faster analysis."
    return im, None, note

def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

def extract_instances(res, ms: float, size: tuple[int, int]) -> dict:
    n = 0 if res.boxes is None else len(res.boxes)
    out = dict(boxes=np.zeros((0, 4), np.float32), cls=np.zeros(0, int), conf=np.zeros(0, np.float32),
               polys=[], ms=ms, size=size)
    if n == 0:
        return out
    boxes = _np(res.boxes.xyxy).astype(np.float32)
    cls = _np(res.boxes.cls).astype(int)
    conf = _np(res.boxes.conf).astype(np.float32)
    polys = [np.asarray(p, dtype=np.float32) for p in res.masks.xy] if getattr(res, "masks", None) is not None else []
    polys += [np.zeros((0, 2), np.float32)] * (n - len(polys))
    order = np.argsort(-conf)
    out.update(boxes=boxes[order], cls=cls[order], conf=conf[order], polys=[polys[i] for i in order])
    return out

@st.cache_data(show_spinner=False, max_entries=16)
def infer(file_hash: str, imgsz: int, _image: Image.Image, _model) -> dict:
    t0 = time.perf_counter()
    res = _model.predict(source=_image, conf=MIN_CONF, imgsz=imgsz, max_det=MAX_DET,
                         retina_masks=True, verbose=False)[0]
    return extract_instances(res, (time.perf_counter() - t0) * 1000.0, _image.size)

def polygon_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()

def render_overlay(img: Image.Image, view: dict, names: dict, show_masks: bool, show_boxes: bool, opacity: float) -> Image.Image:
    base = img.convert("RGBA")
    W, H_ = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    lw = max(2, round(min(W, H_) / 280))
    font = _font(max(12, round(min(W, H_) / 34)))
    alpha = int(255 * opacity)

    items = list(zip(view["polys"], view["boxes"], view["cls"], view["conf"]))
    if show_masks:
        for poly, _, c, _ in items:
            if len(poly) >= 3:
                col = class_info(names.get(int(c), str(c)))["color"]
                d.polygon([tuple(map(float, pt)) for pt in poly], fill=(*col, alpha))
        for poly, _, c, _ in items:
            if len(poly) >= 3:
                col = class_info(names.get(int(c), str(c)))["color"]
                pts = [tuple(map(float, pt)) for pt in poly]
                d.line(pts + [pts[0]], fill=(*col, 255), width=lw, joint="curve")
    if show_boxes:
        for _, box, c, p in items:
            info = class_info(names.get(int(c), str(c)))
            col = info["color"]
            x1, y1, x2, y2 = [float(v) for v in box]
            d.rectangle([x1, y1, x2, y2], outline=(*col, 255), width=lw)
            text = f"{info['short']} {p:.0%}"
            l, t, r, b = d.textbbox((0, 0), text, font=font)
            tw, th = r - l + 12, b - t + 8
            tx = min(max(0, x1), max(0, W - tw))
            ty = y1 - th if y1 - th >= 0 else min(y1, H_ - th)
            d.rectangle([tx, ty, tx + tw, ty + th], fill=(*col, 255))
            lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
            d.text((tx + 6 - l, ty + 4 - t), text, font=font, fill=(16, 32, 26, 255) if lum > 110 else (255, 255, 255, 255))
    return Image.alpha_composite(base, overlay).convert("RGB")

def filter_view(inst: dict, thr: float) -> dict:
    keep = [i for i, c in enumerate(inst["conf"]) if c >= thr]
    return dict(boxes=inst["boxes"][keep], cls=inst["cls"][keep], conf=inst["conf"][keep],
                polys=[inst["polys"][i] for i in keep], ms=inst["ms"], size=inst["size"])

def detections_csv(view: dict, names: dict, filename: str) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "index", "class", "confidence", "x1", "y1", "x2", "y2", "mask_area_px"])
    for i, (c, p, b, poly) in enumerate(zip(view["cls"], view["conf"], view["boxes"], view["polys"]), start=1):
        w.writerow([filename, i, names.get(int(c), str(c)), f"{p:.4f}", *[f"{v:.1f}" for v in b], f"{polygon_area(poly):.0f}"])
    return buf.getvalue().encode("utf-8")

_IMG_EXT = {".png", ".jpg", ".jpeg"}
_PRIORITY = ["results", "confusion_matrix_normalized", "confusion_matrix", "MaskPR_curve", "MaskF1_curve",
             "MaskP_curve", "MaskR_curve", "BoxPR_curve", "val_batch0_pred", "val_batch0_labels", "val_batch1_pred"]
_TEST_RE = re.compile(r"(^|[^a-z])test", re.I)

@st.cache_data(show_spinner=False, ttl=60)
def collect_artifacts() -> tuple[list[str], list[str]]:
    train, test = [], []
    if RUNS_DIR.is_dir():
        for p in RUNS_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in _IMG_EXT:
                folders = p.relative_to(RUNS_DIR).parts[:-1]
                (test if any(_TEST_RE.search(f) for f in folders) else train).append(p)
    key = lambda p: (_PRIORITY.index(p.stem) if p.stem in _PRIORITY else len(_PRIORITY), str(p))
    return [str(p) for p in sorted(train, key=key)], [str(p) for p in sorted(test, key=key)]

_CURVE_KEYS = ("epoch", "train/seg_loss", "val/seg_loss", "train/box_loss", "val/box_loss",
               "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)")

@st.cache_data(show_spinner=False)
def render_curves(hist: dict, best_epoch) -> tuple[bytes, bytes] | None:
    ep = list(hist.get("epoch") or [])
    if not ep:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(key):
        v = list(hist.get(key) or [])
        n = min(len(v), len(ep))
        return ep[:n], v[:n]

    def style(ax, title, ylabel):
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#0F3D24")
        ax.set_xlabel("Epoch", fontsize=9, color="#47594D")
        ax.set_ylabel(ylabel, fontsize=9, color="#47594D")
        ax.grid(color="#E4EFD3", linewidth=0.8)
        ax.set_facecolor("#FFFFFF")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color("#D5E2C0")
        ax.tick_params(colors="#47594D", labelsize=8)
        if best_epoch:
            ax.axvline(best_epoch, color="#FFD43B", linestyle="--", linewidth=1.6, label="Best epoch")
        ax.legend(frameon=False, fontsize=8)

    def to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="white")
    for key, label, color, ls in (("train/seg_loss", "Train mask loss", "#2E7D32", "-"),
                                  ("val/seg_loss", "Val mask loss", "#2E7D32", ":"),
                                  ("train/box_loss", "Train box loss", "#D9A400", "-"),
                                  ("val/box_loss", "Val box loss", "#D9A400", ":")):
        x, y = series(key)
        if y:
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.8, label=label)
    style(ax, "Loss", "Loss")
    loss_png = to_png(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="white")
    for key, label, color in (("metrics/mAP50-95(M)", "mAP@.5:.95", "#1B5E20"), ("metrics/mAP50(M)", "mAP@.5", "#D9A400"),
                              ("metrics/precision(M)", "Precision", "#3BA55D"), ("metrics/recall(M)", "Recall", "#8A6D00")):
        x, y = series(key)
        if y:
            ax.plot(x, y, color=color, linewidth=1.8, label=label)
    style(ax, "Validation metrics (masks)", "Score")
    return loss_png, to_png(fig)

# --------------------------------------------------------------------------- #
# Main App Initialization
# --------------------------------------------------------------------------- #
model_path = find_model_path()
bundle, load_error = None, None

if model_path is None:
    load_error = "No model weights found. Place your trained weights as 'best.pt' next to app.py."
else:
    try:
        bundle = load_model(str(model_path))
    except Exception as e:
        load_error = f"The model could not be loaded: {e}"

if bundle is None:
    st.title("CassavaVision")
    st.error(load_error)
    st.stop()

model, names, imgsz = bundle["model"], bundle["names"], bundle["imgsz"]

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.title("CassavaVision")
    st.caption("Leaf Segmentation")
    st.divider()
    
    st.subheader("Detection")
    conf_thresh = st.slider("Confidence threshold", MIN_CONF, 0.95, 0.35, 0.05)
    
    # Mask opacity placed directly below confidence threshold
    opacity = st.slider("Mask opacity", 0.10, 0.90, 0.45, 0.05)
    
    st.subheader("Display")
    show_masks = st.checkbox("Leaf masks", value=True)
    show_boxes = st.checkbox("Boxes and labels", value=True)
    
    st.divider()
    st.subheader("Classes")
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="height: 12px; width: 12px; background-color: #2EE85C; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span>Healthy</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="height: 12px; width: 12px; background-color: #8B4513; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span>Brown Streak Disease</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="height: 12px; width: 12px; background-color: #DEB823; border-radius: 50%; display: inline-block; margin-right: 8px;"></span>
            <span>Mosaic Disease</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    st.caption(f"Model: {bundle['arch']} - {imgsz}px")

# ---- Main Interface --------------------------------------------------------
st.title("CassavaVision")
st.markdown("Upload a photo of cassava leaves. The model outlines each leaf and flags signs of Cassava Mosaic Disease or Cassava Brown Streak Disease.")

tab_run, tab_model, tab_about = st.tabs(["Analyze", "Model and Results", "About"])

# ---- Analyze Tab -----------------------------------------------------------
with tab_run:
    st.subheader("Upload leaf photos")
    # Restricted to a maximum of 6 images
    files = st.file_uploader("JPG or PNG, up to 10 MB each. Up to 6 images max.", 
                             type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if not files:
        st.markdown("""
        ### How it works
        1. **Upload:** Drop up to 6 leaf photos above.
        2. **Tune:** Adjust the confidence threshold in the sidebar.
        3. **Review:** Check the masks, summary, and table, then download.

        ### For best results
        * Shoot in even daylight, in focus.
        * Fill the frame with the leaf or leaves.
        * Keep the leaf flat and unobstructed.
        * Avoid flash, heavy shadow, and glare.
        """)
    else:
        if len(files) > MAX_FILES:
            st.warning(f"Only the first {MAX_FILES} images are analyzed.")
            files = files[:MAX_FILES]

        # Store batch results for multi-image bottom download functionality
        batch_results = []

        for idx, f in enumerate(files):
            data = f.getvalue()
            st.markdown(f"### {f.name} (Image {idx + 1} of {len(files)})")

            img, err, note = load_image(data)
            if err:
                st.error(err)
                continue

            digest = hashlib.sha1(data).hexdigest()
            try:
                with st.spinner("Analyzing leaves..."):
                    inst = infer(digest, imgsz, img, model)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                continue

            view = filter_view(inst, conf_thresh)
            hidden = len(inst["cls"]) - len(view["cls"])
            
            n = len(view["cls"])
            if n == 0:
                st.info("No leaves detected. Try lowering the confidence threshold.")
            else:
                sick = [c for c in view["cls"] if class_info(names[int(c)])["diseased"]]
                if not sick:
                    st.success(f"{n} leaves detected. All look healthy.")
                else:
                    st.warning(f"{len(sick)} of {n} leaves flagged as diseased.")

            if note:
                st.caption(note)

            n_sick = len([c for c in view["cls"] if class_info(names.get(int(c), str(c)))["diseased"]])
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Leaves Detected", n)
            m2.metric("Flagged Diseased", n_sick)
            m3.metric("Analysis Time", f"{inst['ms']:.0f} ms")

            overlay = render_overlay(img, view, names, show_masks, show_boxes, opacity)
            
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Original")
                show_image(img)
            with c2:
                st.caption("Segmentation")
                show_image(overlay)

            if n:
                st.markdown("#### Detected leaves")
                
                W, H_ = view["size"]
                img_area = float(W * H_) or 1.0
                table_data = []
                for i, (c, p, poly) in enumerate(zip(view["cls"], view["conf"], view["polys"]), start=1):
                    info = class_info(names.get(int(c), str(c)))
                    area = min(100.0, 100.0 * polygon_area(poly) / img_area)
                    table_data.append({
                        "#": i,
                        "Class": info["label"],
                        "Status": "Diseased" if info["diseased"] else "Healthy",
                        "Confidence": f"{p:.1%}",
                        "Image Share": f"{area:.1f}%"
                    })
                st.dataframe(table_data, use_container_width=True)

                buf = io.BytesIO()
                overlay.save(buf, "PNG")
                stem = Path(f.name).stem
                
                # Save data for batch downloading at the bottom
                batch_results.append({
                    "filename": f.name,
                    "stem": stem,
                    "img_bytes": buf.getvalue(),
                    "csv_bytes": detections_csv(view, names, f.name)
                })

                d1, d2 = st.columns(2)
                d1.download_button("Download result image", buf.getvalue(), file_name=f"{stem}_segmented.png", mime="image/png", key=f"dl_img_{idx}_{digest[:8]}")
                d2.download_button("Download detections (CSV)", detections_csv(view, names, f.name), file_name=f"{stem}_detections.csv", mime="text/csv", key=f"dl_csv_{idx}_{digest[:8]}")

            if n == 0:
                st.info("Tip: Ensure the leaf fills the frame and is well-lit. The model only recognizes cassava leaves.")

        # ---- BOTTOM BATCH DOWNLOAD (If more than one image processed) ----
        if len(batch_results) > 1:
            st.divider()
            st.subheader("Batch Downloads")
            st.markdown("Download results for all successfully analyzed images below:")

            import zipfile
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for res in batch_results:
                    zip_file.writestr(f"{res['stem']}_segmented.png", res["img_bytes"])
                    zip_file.writestr(f"{res['stem']}_detections.csv", res["csv_bytes"])
            
            st.download_button(
                label="📦 Download All Results (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="cassavavision_batch_results.zip",
                mime="application/zip",
                key="batch_download_zip"
            )

# ---- Model & Results Tab ---------------------------------------------------
with tab_model:
    args, m = bundle["args"], bundle["metrics"]
    
    st.subheader("Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", bundle["arch"])
    c2.metric("Parameters", f"{bundle['params'] / 1e6:.1f} M" if bundle["params"] else "-")
    c3.metric("Input Size", f"{imgsz} px")
    c4.metric("Classes", str(len(names)))

    st.subheader("Validation Performance")
    pct = lambda k: f"{m[k] * 100:.1f}%" if k in m else "-"
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mask mAP@.5:.95", pct("metrics/mAP50-95(M)"))
    m2.metric("Mask mAP@.5", pct("metrics/mAP50(M)"))
    m3.metric("Precision", pct("metrics/precision(M)"))
    m4.metric("Recall", pct("metrics/recall(M)"))
    m5.metric("Box mAP@.5:.95", pct("metrics/mAP50-95(B)"))

    st.subheader("Learning Curves")
    try:
        subset = {k: list(bundle["history"].get(k) or []) for k in _CURVE_KEYS}
        pngs = render_curves(subset, bundle["best_epoch"])
        if pngs:
            col1, col2 = st.columns(2)
            with col1: show_image(pngs[0])
            with col2: show_image(pngs[1])
        else:
            st.info("Training history not available in this checkpoint.")
    except Exception:
        st.info("Training history not available in this checkpoint.")

    st.subheader("Training Configuration")
    g = lambda k: str(args[k]) if k in args and args[k] is not None else "-"
    config_data = {
        "Max Epochs": g("epochs"),
        "Best Epoch": str(bundle["best_epoch"] or "-"),
        "Batch Size": g("batch"),
        "Optimizer": g("optimizer"),
        "Initial LR": g("lr0"),
        "Patience": g("patience")
    }
    st.table(config_data)

    train_imgs, test_imgs = collect_artifacts()
    
    st.subheader("Training and Validation Artifacts")
    if train_imgs:
        cols = st.columns(2)
        for i, p in enumerate(train_imgs):
            with cols[i % 2]:
                show_image(p, caption=str(Path(p).relative_to(RUNS_DIR)))
    else:
        st.info("No /runs artifacts found. Place your runs folder next to app.py.")

    st.subheader("Held-out Test Set (Unseen Data)")
    if test_imgs:
        cols = st.columns(2)
        for i, p in enumerate(test_imgs):
            with cols[i % 2]:
                show_image(p, caption=str(Path(p).relative_to(RUNS_DIR)))
    else:
        st.info("No test-set artifacts found.")

# ---- About Tab ---------------------------------------------------------------
with tab_about:
    st.subheader("What it does")
    st.markdown("""
    CassavaVision performs **instance segmentation**: it finds each visible cassava leaf, draws a pixel-level mask around it, and labels it as healthy or showing signs of two common diseases.
    Pixel-level masks separate overlapping leaves from cluttered backgrounds, which bounding boxes alone cannot do well.
    """)
    
    st.subheader("Limitations")
    st.markdown("""
    * It only knows cassava leaves; other plants may still be segmented.
    * Unusual lighting, heavy occlusion, or very early symptoms can reduce accuracy.
    * **Results are decision support, not a diagnosis.** Confirm with an agricultural extension officer.
    """)

    st.subheader("Team")
    st.markdown("""
    * Kenneth Ibardaloza
    * Alexis Mesina
    * David Solano
    * Heherson Sarenas
    
    *SOIT, Mapúa University - AI2 Machine Project*
    """)

st.divider()
st.caption("CassavaVision · AI2 Machine Project · Mapúa University · Decision-support prototype.")