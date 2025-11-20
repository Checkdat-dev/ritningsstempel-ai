#import libraries
import os
import cv2
import tempfile
import numpy as np
import fitz  # PyMuPDF

#function to convert pdf to image

def pdf_to_stamp(pdf_bytes: bytes) -> np.ndarray:
    # Save PDF to temporary file (same logic as your code)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        pdf_path = tmp_pdf.name

    try:
       # Using PyMuPDF
       
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)

        # Convert pixmap to numpy image
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # If 4 channels (BGRA), convert to BGR
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Convert to grayscale 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]

        # cropping metadata area from pdf 
        x_start = int(w * 0.76)
        y_start = int(h * 0.88)
        x_end = w
        y_end = int(h * 0.97)

        roi_gray = gray[y_start:y_end, x_start:x_end]
        roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        return roi_bgr

    finally:
        # Close the PDF before deleting
        if doc is not None:
            doc.close()

        # Cleanup temp file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
