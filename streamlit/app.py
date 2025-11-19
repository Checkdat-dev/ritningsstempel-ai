# app.py

import streamlit as st
import pandas as pd
from io import BytesIO
import cv2
import os
import traceback

from utils.pdf_2_image import pdf_to_stamp
from utils.yolo_extract import extract_metadata_from_stamp, ORDER
from utils.clean_up import clean_dataframe


# ------------------------------
# CONFIG
# ------------------------------
PDF_FOLDER = r"D:\Checkdat_LIA\Metadata_extraction\pdf_dataset"
EXCEL_PATH = "extract_all.xlsx"

# FIXED columns (from your clean_up logic)
FIXED_COLUMNS = [
    "AVDELNING",
    "UPPDRAGNUMMER",
    "GODKAND_AV",
    "HANDLINGSTYP",
    "BANDEL",
    "FORMAT",
    "TITEL"
]

def highlight_fixed_cols(row):
    """Highlight fixed columns with yellow background."""
    return ['background-color: #FFF3B0' if col in FIXED_COLUMNS else '' for col in row.index]


# ------------------------------
# STREAMLIT UI
# ------------------------------

st.set_page_config(page_title="Stamp Metadata Extraction", layout="wide")

st.title("📄 Stamp Metadata Extraction (PDF → YOLO → OCR → Cleaning)")

st.write(
    """
This app extracts metadata from the stamp area of a PDF drawing.  
It performs:

1. PDF → cropped stamp (bottom-right)  
2. YOLOv8 detection  
3. EasyOCR extraction  
4. Cleaning (clean_up.py)  
5. Excel output (28 fields)
"""
)


# ============================================================
# 🔍 SEARCH SECTION — Search by filename or RITNINGSNUMMER_PROJEKT
# ============================================================
st.subheader("🔍 Search PDF by Name or RITNINGSNUMMER_PROJEKT")

search_value = st.text_input("Enter File Name OR RITNINGSNUMMER_PROJEKT")


def find_pdf(target):
    """Search by PDF filename or extracted RITNINGSNUMMER_PROJEKT."""
    target = target.lower()

    # Direct filename match
    for f in os.listdir(PDF_FOLDER):
        if target in f.lower() and f.lower().endswith(".pdf"):
            return os.path.join(PDF_FOLDER, f)

    # Search inside extracted Excel
    if os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)

        if "RITNINGSNUMMER_PROJEKT" in df.columns:
            match = df[df["RITNINGSNUMMER_PROJEKT"]
                        .astype(str)
                        .str.lower()
                        .str.contains(target, na=False)]

            if len(match) > 0:
                file_name = match.iloc[0]["filename"]
                return os.path.join(PDF_FOLDER, file_name)

    return None


# ------------------------------------------------------------
# 🔒 STORE SEARCHED PDF IN SESSION STATE
# ------------------------------------------------------------
if "searched_pdf" not in st.session_state:
    st.session_state["searched_pdf"] = None

if st.button("Search"):
    if search_value.strip() == "":
        st.warning("Please enter something to search.")
    else:
        path = find_pdf(search_value.strip())
        if path:
            st.success(f"Found PDF: {path}")
            st.session_state["searched_pdf"] = path
        else:
            st.error("❌ PDF not found.")
            st.session_state["searched_pdf"] = None


# ============================================================
# MERGE SEARCHED PDF INTO `pdf_list`
# ============================================================

uploaded_pdfs = st.file_uploader("Or upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

pdf_list = []

# Add searched PDF if found
if st.session_state["searched_pdf"]:
    with open(st.session_state["searched_pdf"], "rb") as f:
        pdf_bytes = BytesIO(f.read())
        pdf_bytes.name = os.path.basename(st.session_state["searched_pdf"])
        pdf_list.append(pdf_bytes)

# Add uploaded PDFs
if uploaded_pdfs:
    pdf_list.extend(uploaded_pdfs)


# ============================================================
# EXTRACTION PROCESS
# ============================================================
if pdf_list:

    if st.button("🚀 Extract Metadata"):

        all_cleaned_rows = []

        try:
            for pdf in pdf_list:

                st.markdown(f"### 📄 Processing: **{pdf.name}**")

                # STEP 1 — PDF → Stamp
                pdf_bytes = pdf.read()
                stamp_img = pdf_to_stamp(pdf_bytes)

                st.subheader("📌 Cropped Stamp")
                stamp_rgb = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2RGB)
                st.image(stamp_rgb, use_container_width=True)

                # STEP 2 — YOLO + OCR
                raw_dict = extract_metadata_from_stamp(stamp_img)
                raw_dict["filename"] = pdf.name

                raw_df = pd.DataFrame([raw_dict], columns=["filename"] + ORDER)

                st.subheader("📌 Raw Extracted Metadata")
                st.dataframe(raw_df, use_container_width=True)

                # STEP 3 — Cleaning
                cleaned_df = clean_dataframe(raw_df)

                # Highlight fixed columns
                styled_clean = cleaned_df.style.apply(highlight_fixed_cols, axis=1)

                st.subheader("📌 Cleaned Metadata (Fixed Fields Highlighted)")
                st.dataframe(styled_clean, use_container_width=True)

                all_cleaned_rows.append(cleaned_df)

            # STEP 4 — Combine all rows
            final_df = pd.concat(all_cleaned_rows, ignore_index=True)

            # STEP 5 — Download Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                final_df.to_excel(writer, index=False, sheet_name="stamp")
            output.seek(0)

            st.download_button(
                label="💾 Download Full Excel (All PDFs)",
                data=output,
                file_name="all_stamps_cleaned.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error("❌ ERROR OCCURRED")
            st.code(traceback.format_exc())

else:
    st.info("⬆️ Search OR Upload PDFs to start.")

