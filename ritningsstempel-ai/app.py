import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF

from dataextractionsystem import extract_from_pages
from cleanData import clean_data


# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
# Your custom YOLO model lives here:
MODEL_PATH = BASE_DIR / "weights" / "best.pt"  # <-- make sure this file exists in the repo


# ---------- Crop region (stamp area) ----------
LEFT_FRACTION = 0.76
TOP_FRACTION = 0.88
RIGHT_FRACTION = 1.00
BOTTOM_FRACTION = 0.97


def crop_metadata_region(pil_img: Image.Image) -> Image.Image:
    """
    Crop the bottom-right metadata block using fixed fractions of image size.
    """
    W, H = pil_img.size

    left = int(W * LEFT_FRACTION)
    top = int(H * TOP_FRACTION)
    right = int(W * RIGHT_FRACTION)
    bottom = int(H * BOTTOM_FRACTION)

    # Clamp to image bounds
    left = max(0, min(left, W))
    right = max(0, min(right, W))
    top = max(0, min(top, H))
    bottom = max(0, min(bottom, H))

    if right <= left or bottom <= top:
        # fallback: return original if cropping fails
        return pil_img

    return pil_img.crop((left, top, right, bottom))


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame to Excel bytes for download_button.
    """
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")
    return out.getvalue()


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Checkdat – PDF to Metadata", layout="wide")

st.title("📄 CHECKDAT (PDF To Metadata Extraction)")
st.write(
    "Upload a PDF drawing, extract the metadata stamp with YOLO + OCR, "
    "and download both raw and cleaned Excel files."
)

uploaded_pdf = st.file_uploader("Upload a PDF drawing", type=["pdf"])

if uploaded_pdf is not None:
    st.info(f"File uploaded: {uploaded_pdf.name}")

    if st.button("🔍 Run extraction"):
        # Check if YOLO weights exist
        if not MODEL_PATH.exists():
            st.error(
                f"YOLO weights not found at:\n`{MODEL_PATH}`\n\n"
                "Please add your trained model file (e.g. `best.pt`) "
                "to a `weights/` folder in the repo."
            )
            st.stop()

        # 1) PDF -> cropped page images
        st.write("Converting PDF pages to images...")
        file_bytes = uploaded_pdf.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        pages = []
        for i, page in enumerate(doc):
            # Render page at 300 dpi
            pix = page.get_pixmap(dpi=300)
            full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Crop to stamp area
            cropped_img = crop_metadata_region(full_img)

            # Store as (PIL image, page identifier)
            page_name = f"{uploaded_pdf.name}_page_{i+1}"
            pages.append((cropped_img, page_name))

        st.write(f"Found {len(pages)} cropped pages, running YOLO + OCR...")

        # 2) Raw extraction with YOLO + OCR
        with st.spinner("Running YOLO + OCR on stamp regions..."):
            df_raw = extract_from_pages(pages, MODEL_PATH)

        if df_raw.empty:
            st.warning("No metadata could be extracted from this PDF.")
            st.stop()

        st.success("Raw extraction complete!")

        # 3) Cleaning
        with st.spinner("Cleaning data..."):
            df_clean = clean_data(df_raw)

        st.success("Cleaning complete!")

        # 4) Show tables
        st.subheader("📊 Raw data")
        st.dataframe(df_raw, use_container_width=True)

        st.subheader("🧹 Clean data")
        st.dataframe(df_clean, use_container_width=True)

        # 5) Download as Excel
        raw_xlsx = df_to_excel_bytes(df_raw)
        clean_xlsx = df_to_excel_bytes(df_clean)

        st.download_button(
            label="⬇️ Download RAW data (Excel)",
            data=raw_xlsx,
            file_name="rawData.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="⬇️ Download CLEAN data (Excel)",
            data=clean_xlsx,
            file_name="cleanData.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# --------- Simple style tweak ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #C0C0C0 !important;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
