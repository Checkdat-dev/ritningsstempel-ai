import io
import urllib.request
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import cv2
import requests  # << NEW: robust Google Drive download

from dataextractionsystem import extract_from_pages
from cleanData import clean_data

# ---------------------- MODEL PATH & DOWNLOAD ---------------------- #

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "weights" / "best.pt"

# Your Google Drive file ID
FILE_ID = "1Mmb_tS6vyUhOWbgxdjAfPDUoWI2eIYnl"


def download_file_from_google_drive(file_id: str, destination: Path):
    """
    Robust Google Drive downloader that handles the 'too large to scan' confirm token.
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def ensure_model_downloaded():
    """Download YOLO weights from Google Drive if not present."""
    if not MODEL_PATH.exists():
        st.warning("Downloading YOLO model weights from Google Drive... Please wait.")
        download_file_from_google_drive(FILE_ID, MODEL_PATH)
        st.success("Model downloaded!")

    # Debug: show file size so we know it's not just HTML
    size = MODEL_PATH.stat().st_size
    st.write(f"Model file size on disk: {size} bytes")


# ---- Crop region fractions (same as your standalone script) ----
LEFT_FRACTION   = 0.76
TOP_FRACTION    = 0.88
RIGHT_FRACTION  = 1.00
BOTTOM_FRACTION = 0.97


def crop_metadata_region(pil_img: Image.Image) -> Image.Image:
    """Crop the bottom-right metadata block using your fraction rules."""
    W, H = pil_img.size

    left   = int(W * LEFT_FRACTION)
    top    = int(H * TOP_FRACTION)
    right  = int(W * RIGHT_FRACTION)
    bottom = int(H * BOTTOM_FRACTION)

    # Clamp to image bounds
    left   = max(0, min(left, W))
    right  = max(0, min(right, W))
    top    = max(0, min(top, H))
    bottom = max(0, min(bottom, H))

    if right <= left or bottom <= top:
        # fallback: return original if something went weird
        return pil_img

    return pil_img.crop((left, top, right, bottom))


# ---------------------- PAGE CONFIG & CSS ---------------------- #

st.set_page_config(
    page_title="Metadata Extractor",
    page_icon="📄",
    layout="wide",
)

# Simple CSS – silver background
st.markdown(
    """
    <style>
    .stApp {
        background-color: #C0C0C0 !important;  /* Silver */
        color: black;
    }
    .big-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        padding: 0.5rem 0 1rem 0;
        color: #111827;
    }
    .subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .glass-card {
        background: rgba(243, 244, 246, 0.9);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(148, 163, 184, 0.8);
        box-shadow: 0 12px 28px rgba(148, 163, 184, 0.6);
    }
    .metric-card {
        background: rgba(243, 244, 246, 0.95);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        border: 1px solid rgba(148, 163, 184, 0.8);
        text-align: center;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #111827;
    }
    .small-caption {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- SIDEBAR ---------------------- #

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.write("Upload a PDF and run the extractor on its metadata block.")

    uploaded_pdf = st.file_uploader("PDF drawing", type=["pdf"])

    dpi = st.slider("Render DPI", 150, 400, 300, step=50, help="Resolution for PDF → image conversion")
    show_first_crop = st.checkbox("Show first cropped page preview", value=True)

    run_button = st.button("🔍 Run extraction", use_container_width=True)

# ---------------------- MAIN TITLE ---------------------- #

st.markdown('<div class="big-title">Metadata Extraction (YOLO + OCR)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload PDF drawings, auto-detect the metadata block, '
    "run YOLO + EasyOCR, and download both raw and clean Excel outputs.</div>",
    unsafe_allow_html=True,
)

if uploaded_pdf is None:
    st.info("⬅️ Start by uploading a PDF file in the sidebar.")
    st.stop()

if run_button:
    # Make sure model is available
    ensure_model_downloaded()

    with st.spinner("Reading PDF and cropping pages..."):
        file_bytes = uploaded_pdf.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        pages_cv2 = []
        first_cropped_pil = None

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=dpi)
            full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Crop to metadata block
            cropped_img = crop_metadata_region(full_img)

            if i == 0:
                first_cropped_pil = cropped_img

            # PIL -> OpenCV (np.ndarray, BGR)
            img_cv2 = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            pages_cv2.append((img_cv2, f"{uploaded_pdf.name}_page_{i+1}"))

    # Preview first cropped region if requested
    if show_first_crop and first_cropped_pil is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🧩 Cropped metadata block preview (page 1)")
        st.image(first_cropped_pil, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.write(f"Found **{len(pages_cv2)}** cropped pages, running YOLO + OCR...")

    # 2) Extract RAW data
    with st.spinner("Extracting raw metadata..."):
        df_raw = extract_from_pages(pages_cv2, MODEL_PATH)

    if df_raw.empty:
        st.warning("No metadata could be extracted from this PDF.")
        st.stop()

    # 3) Clean data
    with st.spinner("Cleaning data..."):
        df_clean = clean_data(df_raw)

    # ---------------------- SUMMARY METRICS ---------------------- #
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Pages processed</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(pages_cv2)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-caption">From uploaded PDF</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Rows extracted</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df_raw)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-caption">Uncleaned metadata</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Columns</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{len(df_raw.columns)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-caption">(incl. Image)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.success("✅ Extraction and cleaning complete!")

    # ---------------------- TABS: RAW vs CLEAN ---------------------- #
    st.markdown("")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📊 Raw data", "🧹 Clean data"])

    with tab1:
        st.dataframe(df_raw, use_container_width=True, height=420)

    with tab2:
        st.dataframe(df_clean, use_container_width=True, height=420)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- DOWNLOAD BUTTONS ---------------------- #
    def to_excel_bytes(df: pd.DataFrame) -> bytes:
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        return out.getvalue()

    raw_xlsx = to_excel_bytes(df_raw)
    clean_xlsx = to_excel_bytes(df_clean)

    st.markdown("")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="⬇️ Download RAW data (Excel)",
            data=raw_xlsx,
            file_name="rawData.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label="⬇️ Download CLEAN data (Excel)",
            data=clean_xlsx,
            file_name="cleanData.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
