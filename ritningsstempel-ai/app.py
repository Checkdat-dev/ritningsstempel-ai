import io

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
import cv2

from dataextractionsystem import extract_from_pages
from cleanData import clean_data


# Path to your YOLO weights
MODEL_PATH = "weights\best.pt"

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


st.set_page_config(page_title="Metadata Extractor", layout="wide")

st.title("📄 CHECKDAT( PDF To Metadata Extraction)")

uploaded_pdf = st.file_uploader("Upload a PDF drawing", type=["pdf"])

if uploaded_pdf is not None:
    st.info(f"File uploaded: {uploaded_pdf.name}")

    if st.button("🔍 Run extraction"):
        # 1) Convert PDF -> page images
        st.write("Converting PDF pages to images...")
        file_bytes = uploaded_pdf.read()
        doc = fitz.open(stream=file_bytes, filetype="pdf")

        pages_cv2 = []
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # 👉 NEW: crop to your metadata area
            cropped_img = crop_metadata_region(full_img)

            # Optional: preview the first crop for debugging
            # if i == 0:
            #     st.image(cropped_img, caption="Cropped region from first page", use_column_width=True)

            # PIL -> OpenCV (np.ndarray, BGR)
            img_cv2 = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2BGR)
            pages_cv2.append((img_cv2, f"{uploaded_pdf.name}_page_{i+1}"))

        st.write(f"Found {len(pages_cv2)} cropped pages, running YOLO + OCR...")

        # 2) Extract RAW data with your existing logic
        with st.spinner("Extracting raw metadata..."):
            df_raw = extract_from_pages(pages_cv2, MODEL_PATH)

        if df_raw.empty:
            st.warning("No metadata could be extracted from this PDF.")
            st.stop()

        st.success("Raw extraction complete!")

        # 3) Clean the data
        with st.spinner("Cleaning data..."):
            df_clean = clean_data(df_raw)

        st.success("Cleaning complete!")

        # 4) Show tables
        st.subheader("📊 Raw data")
        st.dataframe(df_raw, use_container_width=True)

        st.subheader("🧹 Clean data")
        st.dataframe(df_clean, use_container_width=True)

        # 5) Download buttons (Excel)
        def to_excel_bytes(df: pd.DataFrame) -> bytes:
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Sheet1")
            return out.getvalue()

        raw_xlsx = to_excel_bytes(df_raw)
        clean_xlsx = to_excel_bytes(df_clean)

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
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background-color: #C0C0C0 !important;  /* Silver background */
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
