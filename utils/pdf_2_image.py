import tempfile
import numpy as np
from pdf2image import convert_from_bytes

def pdf_to_stamp(pdf_bytes: bytes) -> np.ndarray:
    # Convert first page to high-resolution image
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    
    if len(pages) == 0:
        raise ValueError("PDF has no pages.")

    # Convert PIL → NumPy BGR
    img = np.array(pages[0])[:, :, ::-1]

    h, w = img.shape[:2]

    # Stamp region cropping (same logic)
    x_start = int(w * 0.76)
    y_start = int(h * 0.88)
    x_end = w
    y_end = int(h * 0.97)

    # Return cropped stamp area
    return img[y_start:y_end, x_start:x_end]
