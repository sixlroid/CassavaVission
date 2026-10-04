"""
CassavaVision - cassava leaf instance segmentation (YOLO26-seg) web app.

AI2 Machine Project - Mapua University
Run with:   streamlit run app.py

Expected layout (all optional except best.pt):
    app.py
    best.pt                      <- trained weights (or runs/**/weights/best.pt)
    runs/                        <- training/validation artifacts (graphs, predictions)
        ...                      <- any sub-folder with "test" in its name is shown as
                                    held-out test-set evidence
"""
from __future__ import annotations

import csv
import hashlib
import html
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
# Page config (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="CassavaVision | Cassava Leaf Segmentation",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
RUNS_DIR = APP_DIR / "runs"

DEFAULT_IMGSZ = 640        # fallback only; the real value is read from the checkpoint
MAX_FILE_MB = 10           # reject uploads larger than this
MAX_FILES = 6              # max images processed per batch (keeps latency reasonable)
MIN_SIDE = 64              # reject images smaller than this (px)
MAX_SIDE = 1600            # larger images are downscaled before inference (px)
MIN_CONF = 0.05            # inference runs once at this floor; the slider filters afterwards
MAX_DET = 100

# name used by the model -> display metadata. Colors are RGB.
CLASS_INFO = {
    "HealthyCassava": dict(
        label="Healthy leaf", short="Healthy", color=(46, 184, 92), diseased=False,
        desc="Uniform green leaf with no mosaic or streak pattern.",
    ),
    "MosaicDisease": dict(
        label="Cassava Mosaic Disease", short="CMD", color=(255, 200, 20), diseased=True,
        desc="Yellow-green mosaic patches, often with curled or distorted leaves.",
    ),
    "BrownStreak": dict(
        label="Cassava Brown Streak Disease", short="CBSD", color=(217, 98, 43), diseased=True,
        desc="Yellow, feathery patches that follow the veins; browning in later stages.",
    ),
}
_FALLBACK_COLORS = [(46, 184, 92), (255, 200, 20), (217, 98, 43), (90, 160, 220), (160, 110, 200)]


def class_info(name: str) -> dict:
    """Display metadata for a model class (robust to unknown class names)."""
    if name in CLASS_INFO:
        return CLASS_INFO[name]
    return dict(
        label=name, short=name,
        color=_FALLBACK_COLORS[zlib.crc32(name.encode()) % len(_FALLBACK_COLORS)],
        diseased="healthy" not in name.lower(), desc="",
    )


# --------------------------------------------------------------------------- #
# Theme (greens + yellows, near-black / white text)
# --------------------------------------------------------------------------- #
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
@media screen {
:root{
 --bg:#F6F8EE; --ink:#10201A; --muted:#47594D; --line:#D5E2C0;
 --g900:#0F3D24; --g700:#1B5E20; --g600:#2E7D32; --g500:#3BA55D; --g100:#E4EFD3; --g50:#F1F6E6;
 --y500:#FFD43B; --y400:#FFE06E; --y100:#FFF4C2; --y700:#B58900;
}
html, body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg); color:var(--ink);
 font-family:'DM Sans',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif; }
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer, [data-testid="stDeployButton"]{ visibility:hidden; }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1200px; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li{ color:var(--ink); }

/* Sidebar */
[data-testid="stSidebar"]{ background:var(--g100); border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{ color:var(--ink); }
.cv-brand{ display:flex; align-items:center; gap:10px; margin:2px 0 14px; }
.cv-brand .logo{ width:36px; height:36px; border-radius:11px; background:var(--g700); display:grid; place-items:center; }
.cv-brand .logo i{ width:14px; height:14px; border-radius:2px 12px 2px 12px; background:var(--y500); display:block; transform:rotate(-12deg); }
.cv-brand b{ font-size:1.15rem; color:var(--g900); letter-spacing:-.01em; }
.cv-brand span{ display:block; font-size:.74rem; color:var(--muted); margin-top:-2px; }
.cv-side-h{ font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--g700); margin:18px 0 4px; }
.cv-rule{ border:0; border-top:1px solid var(--line); margin:16px 0; }
.cv-legend-row{ display:flex; align-items:center; gap:9px; font-size:.88rem; margin:6px 0; color:var(--ink); }
.cv-dot{ width:12px; height:12px; border-radius:50%; display:inline-block; flex:none; box-shadow:0 0 0 2px #fff; }

/* Hero */
.cv-hero{ background:linear-gradient(120deg,#0F3D24 0%,#1B5E20 55%,#3F7D20 100%); border-radius:22px; padding:30px 34px;
 position:relative; overflow:hidden; margin-bottom:18px; }
.cv-hero::after{ content:""; position:absolute; right:-70px; top:-70px; width:280px; height:280px; border-radius:50%;
 background:radial-gradient(circle,rgba(255,212,59,.55),rgba(255,212,59,0) 70%); }
.cv-hero h1{ margin:0; padding:0; font-size:2.1rem; color:#FFFFFF !important; letter-spacing:-.02em; font-weight:700; }
.cv-hero h1 em{ font-style:normal; color:var(--y500) !important; }
.cv-hero p{ color:#E8F3D8 !important; margin:.5rem 0 0; max-width:660px; font-size:1.02rem; line-height:1.5; }
.cv-chips{ margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; position:relative; z-index:1; }
.cv-chip{ background:rgba(255,212,59,.16); border:1px solid rgba(255,212,59,.6); color:#FFE98A; border-radius:999px;
 padding:4px 12px; font-size:.78rem; font-weight:600; }

/* Section titles */
.cv-sec{ display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.08rem; color:var(--g900); margin:26px 0 10px; }
.cv-sec::before{ content:""; width:5px; height:20px; border-radius:3px; background:var(--y500); }
.cv-sub{ color:var(--muted); font-size:.92rem; margin:-4px 0 12px; }

/* Cards */
.cv-card{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:18px 20px; height:100%; }
.cv-card h4{ margin:0 0 8px; font-size:1rem; color:var(--g900); }
.cv-card p, .cv-card li{ font-size:.93rem; line-height:1.55; color:var(--ink); margin:0 0 6px; }
.cv-card ul{ margin:4px 0 0; padding-left:1.1rem; }
.cv-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:14px; }
.cv-step{ display:flex; gap:12px; align-items:flex-start; }
.cv-step .n{ flex:none; width:30px; height:30px; border-radius:50%; background:var(--y500); color:var(--ink); font-weight:700; display:grid; place-items:center; }
.cv-step b{ display:block; color:var(--g900); margin-bottom:2px; }
.cv-step span{ font-size:.9rem; color:var(--muted); }

/* Banner */
.cv-banner{ display:flex; gap:14px; align-items:center; padding:16px 20px; border-radius:16px; border:1px solid var(--line); margin:8px 0 14px; background:#fff; }
.cv-banner .ico{ flex:none; width:44px; height:44px; border-radius:13px; display:grid; place-items:center; font-size:1.3rem; font-weight:700; background:var(--g100); color:var(--g700); }
.cv-banner h3{ margin:0 !important; padding:0 !important; font-size:1.12rem; color:var(--ink); font-weight:700; }
.cv-banner p{ margin:3px 0 0 !important; color:var(--muted) !important; font-size:.92rem; }
.cv-banner.ok{ background:#E6F5E3; border-color:#9AD3A0; } .cv-banner.ok .ico{ background:var(--g600); color:#fff; }
.cv-banner.warn{ background:var(--y100); border-color:#F2D45C; } .cv-banner.warn .ico{ background:var(--y500); color:var(--ink); }
.cv-banner.error{ background:#FFF1E6; border-color:#F0B58A; } .cv-banner.error .ico{ background:#D9622B; color:#fff; }

/* KPI cards */
.cv-kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:6px; }
.cv-kpi{ background:#fff; border:1px solid var(--line); border-top:4px solid var(--g500); border-radius:14px; padding:13px 16px; }
.cv-kpi.y{ border-top-color:var(--y500); }
.cv-kpi .l{ font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:700; }
.cv-kpi .v{ font-size:1.65rem; font-weight:700; color:var(--g700); line-height:1.25; margin-top:2px; }
.cv-kpi .s{ font-size:.8rem; color:var(--muted); }

/* File header, chips, pills */
.cv-file{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin:28px 0 4px; padding-bottom:8px; border-bottom:2px solid var(--g100); }
.cv-file b{ font-size:1.1rem; color:var(--g900); }
.cv-file span{ font-size:.85rem; color:var(--muted); }
.cv-tag{ display:inline-block; font-size:.74rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--g700); background:var(--g100); padding:3px 10px; border-radius:999px; margin-bottom:6px; }
.cv-pill{ display:inline-flex; align-items:center; gap:7px; padding:3px 11px; border-radius:999px; font-size:.82rem; font-weight:600; border:1px solid var(--line); background:#fff; color:var(--ink); }
.cv-pill.ok{ background:#E6F5E3; border-color:#9AD3A0; }
.cv-pill.warn{ background:var(--y100); border-color:#F2D45C; }
.cv-legend{ display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 4px; }

/* Table */
.cv-tablewrap{ overflow-x:auto; border:1px solid var(--line); border-radius:14px; background:#fff; }
.cv-table{ width:100%; border-collapse:collapse; font-size:.92rem; margin:0; }
.cv-table th{ text-align:left; font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:var(--muted) !important;
 background:var(--g50) !important; padding:10px 14px !important; border:0 !important; border-bottom:1px solid var(--line) !important; }
.cv-table td{ padding:10px 14px !important; border:0 !important; border-bottom:1px solid #EEF3E3 !important; color:var(--ink) !important; background:#fff !important; }
.cv-table tr:last-child td{ border-bottom:0 !important; }
.cv-bar{ display:inline-block; vertical-align:middle; width:80px; height:7px; border-radius:99px; background:var(--g100); margin-right:8px; overflow:hidden; }
.cv-bar i{ display:block; height:100%; background:var(--g500); border-radius:99px; }

/* Key/value list */
.cv-kv{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:10px 22px; }
.cv-kv div{ display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-bottom:1px dashed var(--line); font-size:.92rem; }
.cv-kv span{ color:var(--muted); } .cv-kv b{ color:var(--ink); text-align:right; }

/* Streamlit widgets */
[data-testid="stFileUploaderDropzone"]{ background:#FBFDF3; border:2px dashed #7DB37F; border-radius:16px; padding:1.5rem; transition:all .15s; }
[data-testid="stFileUploaderDropzone"]:hover{ background:#FFF9DB; border-color:#D9B200; }
[data-testid="stFileUploaderDropzone"] button{ background:var(--y500); color:var(--ink); border:1px solid #E0B800; border-radius:10px; font-weight:700; }
[data-testid="stFileUploaderDropzone"] button:hover{ background:var(--y400); border-color:#D9B200; color:var(--ink); }
[data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploaderDropzone"] span{ color:var(--muted); }
.stDownloadButton button, .stButton button{ background:var(--y500); color:var(--ink); border:1px solid #E0B800; border-radius:10px; font-weight:700; }
.stDownloadButton button:hover, .stButton button:hover{ background:var(--y400); border-color:#D9B200; color:var(--ink); }
.stDownloadButton button p, .stButton button p{ color:var(--ink) !important; }
[data-testid="stImage"] img{ border-radius:14px; border:1px solid var(--line); }
[data-testid="stImageCaption"]{ color:var(--muted); }
button[data-baseweb="tab"]{ font-weight:600; font-size:.98rem; }
button[data-baseweb="tab"] p{ color:var(--muted); }
button[data-baseweb="tab"][aria-selected="true"] p{ color:var(--g700); }
[data-baseweb="tab-highlight"]{ background-color:var(--y500) !important; height:3px !important; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:14px; background:#fff; }
.cv-foot{ margin-top:38px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem; text-align:center; }
}
"""


def H(markup: str) -> None:
    """Render an HTML snippet. Lines are stripped/joined so Markdown never turns them into a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def section(title: str, sub: str = "") -> None:
    H(f'<div class="cv-sec">{html.escape(title)}</div>')
    if sub:
        H(f'<div class="cv-sub">{html.escape(sub)}</div>')


def banner(kind: str, title: str, text: str = "") -> None:
    icon = {"ok": "&#10003;", "warn": "!", "error": "&#10005;"}.get(kind, "&#8226;")
    body = f"<p>{html.escape(text)}</p>" if text else ""
    H(f'<div class="cv-banner {kind}"><div class="ico">{icon}</div><div><h3>{html.escape(title)}</h3>{body}</div></div>')


def kpis(items: list[tuple[str, str, str, bool]]) -> None:
    """items: (label, value, sub-text, yellow-accent?)"""
    cells = "".join(
        f'<div class="cv-kpi{" y" if y else ""}"><div class="l">{html.escape(l)}</div>'
        f'<div class="v">{html.escape(v)}</div>' + (f'<div class="s">{html.escape(s)}</div>' if s else "") + "</div>"
        for l, v, s, y in items
    )
    H(f'<div class="cv-kpis">{cells}</div>')


def rgb_css(c: tuple[int, int, int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def show_image(img, caption: str | None = None) -> None:
    """st.image that fills its column across Streamlit versions."""
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.image(img, caption=caption, **kwargs)
            return
        except Exception:
            continue


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
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
    from ultralytics import YOLO  # imported here so a missing install gives a friendly message

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


# --------------------------------------------------------------------------- #
# Image validation + inference
# --------------------------------------------------------------------------- #
def load_image(data: bytes) -> tuple[Image.Image | None, str | None, str | None]:
    """Returns (image, error, note). Never raises for bad user input."""
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
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        return None, ("This file could not be read as an image. It may be corrupted or have the wrong "
                      "extension. Please upload a JPG or PNG photo."), None
    if fmt not in {"JPEG", "PNG"}:
        return None, f"Unsupported image format ({fmt or 'unknown'}). Please upload a JPG or PNG photo.", None

    im = ImageOps.exif_transpose(im)  # respect phone-camera rotation
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    im = im.convert("RGB")

    w, h = im.size
    if min(w, h) < MIN_SIDE:
        return None, f"The image is only {w}x{h} px. Please use a photo at least {MIN_SIDE}px on each side.", None
    note = None
    if max(w, h) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        note = f"Large image ({w}x{h}) was resized to {im.size[0]}x{im.size[1]} for faster analysis."
    return im, None, note


def _np(x) -> np.ndarray:
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def extract_instances(res, ms: float, size: tuple[int, int]) -> dict:
    """Convert an Ultralytics Results object into plain numpy data (cacheable, framework-free)."""
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
    """Run the model once per (image, size). Confidence filtering happens afterwards so the slider is instant."""
    t0 = time.perf_counter()
    res = _model.predict(source=_image, conf=MIN_CONF, imgsz=imgsz, max_det=MAX_DET,
                         retina_masks=True, verbose=False)[0]
    return extract_instances(res, (time.perf_counter() - t0) * 1000.0, _image.size)


def polygon_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# --------------------------------------------------------------------------- #
# Rendering (our own overlay so colors match the legend)
# --------------------------------------------------------------------------- #
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_overlay(img: Image.Image, view: dict, names: dict, show_masks: bool, show_boxes: bool,
                   opacity: float) -> Image.Image:
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


# --------------------------------------------------------------------------- #
# Result interpretation
# --------------------------------------------------------------------------- #
def plural(n: int, word: str) -> str:
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word[:-1] + 'ves' if word.endswith('leaf') else word + 's'}"


def filter_view(inst: dict, thr: float) -> dict:
    keep = [i for i, c in enumerate(inst["conf"]) if c >= thr]
    return dict(boxes=inst["boxes"][keep], cls=inst["cls"][keep], conf=inst["conf"][keep],
                polys=[inst["polys"][i] for i in keep], ms=inst["ms"], size=inst["size"])


def assess(view: dict, names: dict, thr: float, hidden: int) -> tuple[str, str, str]:
    """(banner kind, title, text)"""
    n = len(view["cls"])
    if n == 0:
        text = f"No leaf reached the {thr:.0%} confidence threshold."
        if hidden:
            text += f" {plural(hidden, 'lower-confidence candidate')} hidden - try lowering the threshold."
        return "neutral", "No leaves detected", text
    infos = [class_info(names.get(int(c), str(c))) for c in view["cls"]]
    sick = [(i, p) for i, p in zip(infos, view["conf"]) if i["diseased"]]
    if not sick:
        return "ok", "Leaves look healthy", f"{plural(n, 'leaf')} detected, none showing disease signs."
    kinds = Counter(i["label"] for i, _ in sick)
    if len(kinds) == 1:
        title = f"Signs of {next(iter(kinds))}"
    else:
        title = "Mixed disease signals: " + " + ".join(sorted({i["short"] for i, _ in sick}))
    mean_sick = float(np.mean([p for _, p in sick]))
    noun = "leaf" if n == 1 else "leaves"
    return "warn", title, f"{len(sick)} of {n} detected {noun} flagged as diseased (average confidence {mean_sick:.0%})."


def detections_table(view: dict, names: dict) -> str:
    W, H_ = view["size"]
    img_area = float(W * H_) or 1.0
    rows = []
    for i, (c, p, poly) in enumerate(zip(view["cls"], view["conf"], view["polys"]), start=1):
        info = class_info(names.get(int(c), str(c)))
        status = ('<span class="cv-pill warn">Diseased</span>' if info["diseased"]
                  else '<span class="cv-pill ok">Healthy</span>')
        area = min(100.0, 100.0 * polygon_area(poly) / img_area)
        rows.append(
            f"<tr><td>{i}</td>"
            f'<td><span class="cv-dot" style="background:{rgb_css(info["color"])}"></span> {html.escape(info["label"])}</td>'
            f"<td>{status}</td>"
            f'<td><span class="cv-bar"><i style="width:{p * 100:.0f}%"></i></span>{p:.1%}</td>'
            f"<td>{area:.1f}%</td></tr>"
        )
    head = "<tr><th>#</th><th>Class</th><th>Status</th><th>Confidence</th><th>Share of image</th></tr>"
    return f'<div class="cv-tablewrap"><table class="cv-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def detections_csv(view: dict, names: dict, filename: str) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "index", "class", "confidence", "x1", "y1", "x2", "y2", "mask_area_px"])
    for i, (c, p, b, poly) in enumerate(zip(view["cls"], view["conf"], view["boxes"], view["polys"]), start=1):
        w.writerow([filename, i, names.get(int(c), str(c)), f"{p:.4f}", *[f"{v:.1f}" for v in b], f"{polygon_area(poly):.0f}"])
    return buf.getvalue().encode("utf-8")


def tips_for(view: dict, hidden: int, thr: float) -> list[str]:
    tips = []
    n = len(view["cls"])
    if n == 0:
        if hidden and thr > MIN_CONF + 1e-9:
            tips.append("Lower the confidence threshold in the sidebar to reveal weaker candidates.")
        tips += ["Use a close-up of one or a few leaves, filling most of the frame.",
                 "Photograph in even daylight; avoid heavy shadows, blur or glare.",
                 "The model only knows cassava leaves - other plants may not be recognised."]
    elif float(np.mean(view["conf"])) < 0.5:
        tips += ["Confidence is low overall. Retake the photo with better lighting and focus.",
                 "Make sure the leaf is flat, in frame, and not covered by other foliage."]
    return tips


# --------------------------------------------------------------------------- #
# Run artifacts (/runs folder)
# --------------------------------------------------------------------------- #
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


def artifact_grid(paths: list[str], limit: int = 6) -> None:
    def draw(chunk):
        cols = st.columns(2)
        for i, p in enumerate(chunk):
            with cols[i % 2]:
                show_image(p, caption=str(Path(p).relative_to(RUNS_DIR)))
    draw(paths[:limit])
    if len(paths) > limit:
        with st.expander(f"Show {len(paths) - limit} more"):
            draw(paths[limit:])


# --------------------------------------------------------------------------- #
# Training curves (from the checkpoint, so they work even without /runs)
# --------------------------------------------------------------------------- #
_CURVE_KEYS = ("epoch", "train/seg_loss", "val/seg_loss", "train/box_loss", "val/box_loss",
               "metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)")


@st.cache_data(show_spinner=False)
def render_curves(hist: dict, best_epoch) -> tuple[bytes, bytes] | None:
    """Draw the loss / metric curves once and cache them as PNG bytes (tabs re-run on every widget change)."""
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


def plot_history(meta: dict) -> bool:
    subset = {k: list(meta["history"].get(k) or []) for k in _CURVE_KEYS}
    pngs = render_curves(subset, meta["best_epoch"])
    if not pngs:
        return False
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        show_image(pngs[0])
    with c2:
        show_image(pngs[1])
    return True


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
H(f"<style>{CSS}</style>")


def hero(chips: list[str]) -> None:
    chip_html = "".join(f'<span class="cv-chip">{html.escape(c)}</span>' for c in chips)
    H(f"""
    <div class="cv-hero">
      <h1>Cassava<em>Vision</em></h1>
      <p>Upload a photo of cassava leaves. The model outlines each leaf and flags signs of
      Cassava Mosaic Disease or Cassava Brown Streak Disease.</p>
      <div class="cv-chips">{chip_html}</div>
    </div>""")


def footer() -> None:
    H('<div class="cv-foot">CassavaVision &middot; AI2 Machine Project &middot; Mapua University &middot; '
      "Decision-support prototype - confirm findings with an agricultural expert.</div>")


model_path = find_model_path()
bundle, load_error = None, None
if model_path is None:
    load_error = ("No model weights found. Place your trained weights as best.pt next to app.py "
                  "(or under runs/<run>/weights/best.pt).")
else:
    try:
        bundle = load_model(str(model_path))
    except ImportError as e:
        load_error = f"A required library is missing ({e}). Run: pip install -r requirements.txt"
    except Exception as e:  # corrupted file, version mismatch, ...
        load_error = (f"The model could not be loaded: {e}. Make sure Ultralytics is up to date "
                      "(pip install -U ultralytics) and that best.pt is the file produced by training.")

if bundle is None:
    hero([])
    banner("error", "The model isn't available", load_error or "")
    footer()
    st.stop()

model, names, imgsz = bundle["model"], bundle["names"], bundle["imgsz"]

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    H('<div class="cv-brand"><div class="logo"><i></i></div><div><b>CassavaVision</b><span>Leaf segmentation</span></div></div>')
    H('<div class="cv-side-h">Detection</div>')
    conf_thresh = st.slider(
        "Confidence threshold", MIN_CONF, 0.95, 0.35, 0.05,
        help="Only show leaves the model is at least this sure about. Raise it to remove false alarms; "
             "lower it to catch faint or partly hidden leaves.")
    H('<div class="cv-side-h">Display</div>')
    show_masks = st.checkbox("Leaf masks", value=True)
    show_boxes = st.checkbox("Boxes and labels", value=True)
    opacity = st.slider("Mask opacity", 0.10, 0.90, 0.45, 0.05, disabled=not show_masks)
    H('<hr class="cv-rule"><div class="cv-side-h">Classes</div>')
    for n_ in sorted(names):
        info = class_info(names[n_])
        H(f'<div class="cv-legend-row"><span class="cv-dot" style="background:{rgb_css(info["color"])}"></span>{html.escape(info["label"])}</div>')
    H('<hr class="cv-rule">')
    st.caption(f"Model: {bundle['arch']} - {imgsz}px input")

hero(["Instance segmentation", f"{len(names)} classes", f"{bundle['arch']} - {imgsz}px"])

tab_run, tab_model, tab_about = st.tabs(["Analyze", "Model and results", "About"])

# ---- Analyze tab -----------------------------------------------------------
with tab_run:
    section("Upload leaf photos", f"JPG or PNG, up to {MAX_FILE_MB} MB each. You can select several images at once.")
    files = st.file_uploader("Upload cassava leaf photos", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, label_visibility="collapsed")

    if not files:
        H("""
        <div class="cv-grid" style="margin-top:14px">
          <div class="cv-card"><h4>How it works</h4>
            <div class="cv-step"><div class="n">1</div><div><b>Upload</b><span>Drop one or more leaf photos above.</span></div></div><br>
            <div class="cv-step"><div class="n">2</div><div><b>Tune</b><span>Adjust the confidence threshold in the sidebar.</span></div></div><br>
            <div class="cv-step"><div class="n">3</div><div><b>Review</b><span>Check the masks, summary and per-leaf table, then download.</span></div></div>
          </div>
          <div class="cv-card"><h4>For best results</h4>
            <ul><li>Shoot in even daylight, in focus.</li>
            <li>Fill the frame with the leaf or leaves.</li>
            <li>Keep the leaf flat and unobstructed.</li>
            <li>Avoid flash, heavy shadow and glare.</li></ul>
          </div>
        </div>""")
    else:
        if len(files) > MAX_FILES:
            banner("warn", f"Only the first {MAX_FILES} images are analyzed",
                   f"You uploaded {len(files)}. Remove some to analyze different ones.")
            files = files[:MAX_FILES]

        for idx, f in enumerate(files):
            data = f.getvalue()
            H(f'<div class="cv-file"><b>{html.escape(f.name)}</b><span>Image {idx + 1} of {len(files)}</span></div>')

            img, err, note = load_image(data)
            if err:
                banner("error", "This image can't be analyzed", err)
                continue

            digest = hashlib.sha1(data).hexdigest()
            try:
                with st.spinner("Analyzing leaves... (the first run can take a few seconds)"):
                    inst = infer(digest, imgsz, img, model)
            except Exception as e:
                banner("error", "Analysis failed", "Something went wrong while running the model on this image. "
                       "Try a different photo, or re-upload it.")
                with st.expander("Technical details"):
                    st.code(f"{type(e).__name__}: {e}")
                continue

            view = filter_view(inst, conf_thresh)
            hidden = len(inst["cls"]) - len(view["cls"])
            kind, title, text = assess(view, names, conf_thresh, hidden)
            banner(kind, title, text)
            if note:
                st.caption(note)

            n = len(view["cls"])
            n_sick = sum(class_info(names.get(int(c), str(c)))["diseased"] for c in view["cls"])
            kpis([
                ("Leaves detected", str(n), "", False),
                ("Flagged diseased", f"{n_sick}", f"of {n}" if n else "", n_sick > 0),
                ("Average confidence", f"{float(np.mean(view['conf'])):.0%}" if n else "-", "", False),
                ("Analysis time", f"{inst['ms']:.0f} ms", f"{imgsz}px inference", False),
            ])

            overlay = render_overlay(img, view, names, show_masks, show_boxes, opacity)
            c1, c2 = st.columns(2, gap="medium")
            with c1:
                H('<span class="cv-tag">Original</span>')
                show_image(img)
            with c2:
                H('<span class="cv-tag">Segmentation</span>')
                show_image(overlay)

            if n:
                counts = Counter(int(c) for c in view["cls"])
                chips = "".join(
                    f'<span class="cv-pill"><span class="cv-dot" style="background:{rgb_css(class_info(names.get(c, str(c)))["color"])}"></span>'
                    f'{html.escape(class_info(names.get(c, str(c)))["label"])} &middot; {k}</span>'
                    for c, k in sorted(counts.items()))
                H(f'<div class="cv-legend">{chips}</div>')
                section("Detected leaves")
                H(detections_table(view, names))

                buf = io.BytesIO()
                overlay.save(buf, "PNG")
                stem = Path(f.name).stem
                d1, d2, _ = st.columns([1, 1, 2])
                d1.download_button("Download result image", buf.getvalue(), file_name=f"{stem}_segmented.png",
                                   mime="image/png", key=f"dl_img_{idx}_{digest[:8]}")
                d2.download_button("Download detections (CSV)", detections_csv(view, names, f.name),
                                   file_name=f"{stem}_detections.csv", mime="text/csv", key=f"dl_csv_{idx}_{digest[:8]}")

            tips = tips_for(view, hidden, conf_thresh)
            if tips:
                H('<div class="cv-card" style="margin-top:14px"><h4>Suggestions</h4><ul>'
                  + "".join(f"<li>{html.escape(t)}</li>" for t in tips) + "</ul></div>")

# ---- Model & results tab ---------------------------------------------------
with tab_model:
    args, m = bundle["args"], bundle["metrics"]
    section("Model", "Loaded from the trained checkpoint.")
    kpis([
        ("Architecture", bundle["arch"], "instance segmentation", False),
        ("Parameters", f"{bundle['params'] / 1e6:.1f} M" if bundle["params"] else "-", f"{bundle['size_mb']:.0f} MB weights", False),
        ("Input size", f"{imgsz} px", "train and inference", False),
        ("Classes", str(len(names)), ", ".join(class_info(names[i])["short"] for i in sorted(names)), True),
    ])

    section("Validation performance",
            "Best checkpoint, scored on the validation split during training - not the held-out test set.")
    pct = lambda k: f"{m[k] * 100:.1f}%" if k in m else "-"
    kpis([
        ("Mask mAP@.5:.95", pct("metrics/mAP50-95(M)"), "", True),
        ("Mask mAP@.5", pct("metrics/mAP50(M)"), "", False),
        ("Precision", pct("metrics/precision(M)"), "", False),
        ("Recall", pct("metrics/recall(M)"), "", False),
        ("Box mAP@.5:.95", pct("metrics/mAP50-95(B)"), "", False),
    ])
    if not m:
        st.caption("Metrics were not stored in this checkpoint - see the notebook and /runs artifacts.")

    section("Learning curves", "Plotted from the training history stored in the checkpoint.")
    try:
        ok = plot_history(bundle)
    except Exception:
        ok = False
    if not ok:
        banner("neutral", "Training history not available in this checkpoint", "See the graphs in your /runs folder below.")
    else:
        h = bundle["history"]
        mk, ep = h.get("metrics/mAP50-95(M)") or [], h.get("epoch") or []
        if mk and ep:
            st.caption(f"Mask mAP@.5:.95 moved from {mk[0]:.3f} (epoch {ep[0]}) to {mk[-1]:.3f} (epoch {ep[min(len(mk), len(ep)) - 1]}); "
                       "the dashed line marks the best epoch.")

    section("Training configuration")
    g = lambda k, f="{}": (f.format(args[k]) if k in args and args[k] is not None else "-")
    aug = ", ".join(x for x in (
        f"mosaic {args['mosaic']}" if "mosaic" in args else "", f"flip-LR {args['fliplr']}" if "fliplr" in args else "",
        f"HSV {args.get('hsv_h')}/{args.get('hsv_s')}/{args.get('hsv_v')}" if "hsv_h" in args else "",
        f"scale {args['scale']}" if "scale" in args else "", "RandAugment" if args.get("auto_augment") == "randaugment" else "") if x) or "-"
    rows = [("Max epochs", g("epochs")), ("Best epoch", str(bundle["best_epoch"] or "-")), ("Batch size", g("batch")),
            ("Optimizer", g("optimizer")), ("Initial learning rate", g("lr0")), ("Final LR fraction", g("lrf")),
            ("Momentum", g("momentum")), ("Weight decay", g("weight_decay")), ("Early-stopping patience", g("patience")),
            ("Augmentation", aug)]
    H('<div class="cv-kv">' + "".join(
        f'<div{" style=\'grid-column:1/-1\'" if k == "Augmentation" else ""}><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>'
        for k, v in rows) + "</div>")

    train_imgs, test_imgs = collect_artifacts()
    section("Training and validation artifacts", "Graphs and predictions from your /runs folder.")
    if train_imgs:
        artifact_grid(train_imgs)
    else:
        banner("neutral", "No /runs artifacts found",
               f"Copy your runs folder next to app.py ({RUNS_DIR.name}/) to show confusion matrices, PR curves and validation predictions here.")

    section("Held-out test set (unseen data)", "Evidence from images never used for training or validation.")
    if test_imgs:
        artifact_grid(test_imgs)
    else:
        banner("neutral", "No test-set artifacts found",
               "Put the output folder from your test evaluation (any folder under runs/ with 'test' in its name) "
               "and its graphs and predictions will appear here.")

# ---- About tab ---------------------------------------------------------------
with tab_about:
    section("About this project")
    H("""
    <div class="cv-grid">
      <div class="cv-card"><h4>What it does</h4>
        <p>CassavaVision performs <b>instance segmentation</b>: it finds each visible cassava leaf, draws a pixel-level
        mask around it, and labels it as healthy or as showing signs of one of two common diseases.</p>
        <p>Pixel-level masks separate overlapping leaves from cluttered field backgrounds, which bounding boxes alone cannot do well.</p>
      </div>
      <div class="cv-card"><h4>Classes</h4>""" + "".join(
        f'<p><span class="cv-dot" style="background:{rgb_css(class_info(names[i])["color"])}"></span> '
        f'<b>{html.escape(class_info(names[i])["label"])}</b><br>{html.escape(class_info(names[i])["desc"])}</p>'
        for i in sorted(names)) + """
      </div>
      <div class="cv-card"><h4>Limitations</h4>
        <ul><li>It only knows cassava leaves; other plants or objects may still be segmented.</li>
        <li>Unusual lighting, heavy occlusion or very early symptoms can reduce accuracy.</li>
        <li>Results are decision support, not a diagnosis. Confirm with an agricultural extension officer or lab test.</li></ul>
      </div>
      <div class="cv-card"><h4>Team</h4>
        <p>Kenneth Ibardaloza<br>Alexis Mesina<br>David Solano<br>Heherson Sarenas</p>
        <p style="color:var(--muted)">SOIT, Mapua University - AI2 Machine Project</p>
      </div>
    </div>""")

footer()