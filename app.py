"""
🐍 Snake Classifier — Streamlit Frontend
EfficientNetV2-L | 25 Species | 87.38% TTA Accuracy
"""

import streamlit as st
import torch
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import gdown
import os
import json
import tempfile
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐍 Snake Classifier",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.main {
    background: #0d0d0d;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
}

.stApp {
    background: linear-gradient(135deg, #0d0d0d 0%, #111a0f 50%, #0d0d0d 100%);
}

/* Title banner */
.title-block {
    background: linear-gradient(90deg, #1a2e14 0%, #0d1a0a 100%);
    border: 1px solid #3a7d1e;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.title-block::before {
    content: "🐍";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
}
.title-block h1 {
    color: #7dde3c !important;
    font-size: 2.4rem !important;
    margin: 0 !important;
    letter-spacing: -1px;
}
.title-block p {
    color: #89a87b;
    margin: 0.4rem 0 0 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #141a10;
    border: 1px solid #2a4a1a;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    flex: 1;
    text-align: center;
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #7dde3c;
    font-family: 'Space Mono', monospace;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #89a87b;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Result card */
.result-card {
    background: #141a10;
    border: 2px solid #3a7d1e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.result-species {
    font-size: 1.6rem;
    font-weight: 800;
    color: #7dde3c;
    margin-bottom: 0.3rem;
}
.result-confidence {
    font-family: 'Space Mono', monospace;
    color: #89a87b;
    font-size: 0.9rem;
}

/* Danger / safe badges */
.badge-danger {
    background: #3d0f0f;
    color: #ff6b6b;
    border: 1px solid #a83030;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
    margin-left: 0.5rem;
}
.badge-safe {
    background: #0f2d0f;
    color: #7dde3c;
    border: 1px solid #3a7d1e;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    display: inline-block;
    margin-left: 0.5rem;
}

/* Upload zone */
.upload-hint {
    color: #89a87b;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    margin-top: 0.5rem;
}

/* Footer */
.footer {
    text-align: center;
    color: #3a4a2a;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #1a2a10;
}

div[data-testid="stSidebar"] {
    background: #0d150a;
    border-right: 1px solid #1e3a12;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────
GDRIVE_ID       = "1EHdaSKNH8ZwuY61wQQgFeVsucyy2f2d4"
MODEL_PATH      = "snake_classifier_final.pth"
IMG_SIZE        = 384
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]

# Which species are venomous (rough guide — adjust as needed)
VENOMOUS = {
    "blunt_nosed_viper", "caspian_cobra", "common_krait", "himalayan_pit_viper",
    "indian_cobra", "russell_s_vipe", "saharan_horned_viper", "saw_scaled_viper",
    "spectacled_cobra",
}

# ─── TTA Transforms ──────────────────────────────────────────────────────────
try:
    import torchvision.transforms as T
    TTA_TFS = [
        T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),                              T.ToTensor(), T.Normalize(MEAN, STD)]),
        T.Compose([T.Resize((IMG_SIZE+32, IMG_SIZE+32)), T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)]),
        T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize(MEAN, STD)]),
        T.Compose([T.Resize((IMG_SIZE+32, IMG_SIZE+32)), T.RandomCrop(IMG_SIZE),  T.ToTensor(), T.Normalize(MEAN, STD)]),
        T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.RandomRotation(10),           T.ToTensor(), T.Normalize(MEAN, STD)]),
    ]
    VAL_TF = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
except Exception:
    TTA_TFS = None
    VAL_TF  = None

# ─── Model loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Download from Google Drive if needed, then load checkpoint."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️  Downloading model from Google Drive (~450 MB)…"):
            url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    arch   = ckpt.get("architecture", "tf_efficientnetv2_l")
    nc     = ckpt["num_classes"]

    model = timm.create_model(arch, pretrained=False, num_classes=nc, drop_rate=0.3)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    idx_to_class = ckpt["idx_to_class"]
    return model, idx_to_class, device

# ─── Inference ───────────────────────────────────────────────────────────────
def predict(img_pil: Image.Image, model, idx_to_class, device, use_tta: bool = True):
    if use_tta and TTA_TFS:
        probs_list = []
        for tf in TTA_TFS:
            inp = tf(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                probs_list.append(torch.softmax(model(inp), dim=1).cpu())
        probs = torch.stack(probs_list).mean(0)[0]
    else:
        inp = VAL_TF(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0].cpu()

    top5 = probs.topk(5)
    results = []
    for prob, idx in zip(top5.values, top5.indices):
        key     = idx_to_class[idx.item()]
        display = key.replace("_", " ").title()
        results.append({
            "key":        key,
            "species":    display,
            "confidence": prob.item(),
            "venomous":   key in VENOMOUS,
        })
    return results

# ─── Chart helper ────────────────────────────────────────────────────────────
def make_chart(results):
    species = [r["species"][:32] for r in reversed(results)]
    confs   = [r["confidence"] * 100 for r in reversed(results)]
    colors  = ["#ff6b6b" if results[4 - i]["venomous"] else "#7dde3c" for i in range(5)]

    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("#141a10")
    ax.set_facecolor("#141a10")

    bars = ax.barh(species, confs, color=colors, height=0.55, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", color="#89a87b", fontsize=8)
    ax.tick_params(colors="#89a87b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a4a1a")
    ax.xaxis.label.set_color("#89a87b")
    ax.yaxis.set_tick_params(labelcolor="#c8e6a0")

    for bar, conf in zip(bars, confs):
        ax.text(conf + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{conf:.1f}%", va="center", fontsize=7.5,
                color="#ffffff", fontweight="bold")

    red_patch   = mpatches.Patch(color="#ff6b6b", label="Venomous")
    green_patch = mpatches.Patch(color="#7dde3c", label="Non-venomous")
    ax.legend(handles=[red_patch, green_patch], loc="lower right",
              fontsize=7, framealpha=0, labelcolor="#89a87b")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    use_tta = st.toggle("Use TTA (5 views)", value=True,
                        help="Test-Time Augmentation — averages 5 predictions for higher accuracy. Slightly slower.")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    | | |
    |---|---|
    | **Architecture** | EfficientNetV2-L |
    | **Parameters** | 117M |
    | **Image size** | 384 × 384 |
    | **Classes** | 25 species |
    | **Test accuracy** | **87.38%** (TTA) |
    | **Loss** | Focal + Label Smoothing |
    """)

    st.markdown("---")
    st.markdown("### 🔴 Venomous Species")
    venomous_display = [v.replace("_", " ").title() for v in sorted(VENOMOUS)]
    for v in venomous_display:
        st.markdown(f"- {v}")

    st.markdown("---")
    st.caption("Dataset: 21,292 images | South Asian & Middle Eastern snakes")

# ─── Main UI ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>Snake Classifier</h1>
  <p>EfficientNetV2-L · TTA · Mixup · Focal Loss · 25 Species · 87.38% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# Metric row
st.markdown("""
<div class="metric-row">
  <div class="metric-card"><div class="val">87.38%</div><div class="lbl">TTA Accuracy</div></div>
  <div class="metric-card"><div class="val">25</div><div class="lbl">Species</div></div>
  <div class="metric-card"><div class="val">117M</div><div class="lbl">Parameters</div></div>
  <div class="metric-card"><div class="val">5×</div><div class="lbl">TTA Views</div></div>
</div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("🔄 Loading model…"):
    try:
        model, idx_to_class, device = load_model()
        st.success(f"✅ Model loaded — running on **{device}**")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

st.markdown("---")

# Upload
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📸 Upload Snake Image")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )
    st.markdown('<p class="upload-hint">Supported: JPG · PNG · WEBP · BMP</p>', unsafe_allow_html=True)

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded image", use_container_width=True)

with col2:
    st.markdown("### 🔍 Prediction")
    if not uploaded:
        st.info("⬅️  Upload a snake image to classify it.")
    else:
        with st.spinner(f"{'🔬 Running TTA inference (5 views)…' if use_tta else '⚡ Running inference…'}"):
            t0      = time.time()
            results = predict(img_pil, model, idx_to_class, device, use_tta=use_tta)
            elapsed = time.time() - t0

        top = results[0]
        badge = '<span class="badge-danger">⚠️ VENOMOUS</span>' if top["venomous"] \
                else '<span class="badge-safe">✔ NON-VENOMOUS</span>'

        st.markdown(f"""
        <div class="result-card">
          <div class="result-species">{top['species']}{badge}</div>
          <div class="result-confidence">Confidence: {top['confidence']*100:.1f}% &nbsp;|&nbsp; Inference: {elapsed:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Top-5 Predictions")
        chart_buf = make_chart(results)
        st.image(chart_buf, use_container_width=True)

        with st.expander("📋 Full top-5 table"):
            for i, r in enumerate(results, 1):
                icon = "🔴" if r["venomous"] else "🟢"
                st.markdown(
                    f"**#{i}** {icon} `{r['species']}` — **{r['confidence']*100:.2f}%**"
                )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Snake Classifier v2 · EfficientNetV2-L · Built with Streamlit · ⚠️ For educational use only — always consult an expert for snake identification.
</div>
""", unsafe_allow_html=True)
