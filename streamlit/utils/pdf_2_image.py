import os
import cv2
import tempfile
import numpy as np
from pdf2image import convert_from_path

POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

def pdf_to_stamp(pdf_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_bytes)
        pdf_path = tmp_pdf.name

    temp_png = None

    try:
        pages = convert_from_path(
            pdf_path,
            dpi=300,
            poppler_path=POPPLER_PATH
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
            temp_png = tmp_png.name
            pages[0].save(temp_png, "PNG")

        img = cv2.imread(temp_png)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]

        x_start = int(w * 0.76)
        y_start = int(h * 0.88)
        x_end = w
        y_end = int(h * 0.97)

        roi_gray = gray[y_start:y_end, x_start:x_end]
        roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        return roi_bgr

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if temp_png and os.path.exists(temp_png):
            os.remove(temp_png)
