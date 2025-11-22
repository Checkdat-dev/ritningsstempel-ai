import numpy as np
import tempfile

def pdf_to_stamp(pdf_bytes: bytes) -> np.ndarray:
    # Lazy import (Streamlit Cloud safe)
    import fitz  # PyMuPDF

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        pdf_path = tmp_pdf.name

    # Open PDF using PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)

    # Render page to image (high DPI)
    pix = page.get_pixmap(dpi=300)

    # Convert pixmap to numpy array (RGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)  # pix.n is number of channels

    # Convert RGBA → RGB if needed
    if pix.n == 4:
        img = img[:, :, :3]

    h, w = img.shape[:2]

    # Stamp crop area
    x_start = int(w * 0.76)
    y_start = int(h * 0.88)
    x_end = w
    y_end = int(h * 0.97)

    stamp = img[y_start:y_end, x_start:x_end]

    doc.close()
    return stamp
