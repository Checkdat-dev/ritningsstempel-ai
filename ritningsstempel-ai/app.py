import io
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import fitz  # PyMuPDF
from dataextractionsystem import extract_from_pages
from cleanData import clean_data


# ---------- Paths ----------
# Assume your YOLO weights are stored in:  models/best.pt  (relative to this file)
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"   # <--- put your custom YOLO weights here


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


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Checkdat – PDF to Metadata", layout="wide")

st.title("📄 CHECKDAT (PDF To Metadata Extraction)")
st.write("Upload a PDF drawing, extract the metadata stamp with YOLO + OCR, and download clean Excel.")


uploaded_pdf = st.file_uploader("Upload a PDF drawing", type=["pdf"])

if uploaded_pdf is not None:
    st.info(f"File uploaded: {uploaded_pdf.name}")

    if st.button("🔍 Run extraction"):
        if not MODEL_PATH.exists():
            st.error(
                f"YOLO weights not found at:\n`{MODEL_PATH}`\n\n"
                "Please add your trained model file (e.g. `best.pt`) "
                "to a `models/` folder in the repo."
            )
        else:
            # 1) PDF -> cropped page images
            st.write("Converting PDF pages to images...")
            file_bytes = uploaded_pdf.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            pages = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                full_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                cropped_img = crop_metadata_region(full_img)

                # Store as PIL image; dataextractionsystem can handle PIL or NumPy
                pages.append((cropped_img, f"{uploaded_pdf.name}_page_{i+1}"))

            st.write(f"Found {len(pages)} cropped pages, running YOLO + OCR...")

            # 2) Raw extraction
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

# Simple style tweak (silver background like before)
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
