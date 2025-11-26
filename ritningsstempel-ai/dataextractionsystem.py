from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image
import easyocr
from ultralytics import YOLO


# --------------------------------------------------
# Labels detected by YOLO and returned in the Excel
# --------------------------------------------------
LABELS: List[str] = [
    "Anläggningstyp",
    "Avdelning",
    "Bandel",
    "Beskrivning 1",
    "Beskrivning 2",
    "Blad",
    "Datum",
    "Format",
    "Godkänd Av",
    "Granskad Av",
    "Granskningsstatus/Syfte",
    "Handlingstyp",
    "Kilometer & Meter",
    "Leverans/Ändrings-pm",
    "Leverantör",
    "Nästa Blad",
    "Projektsnamn",
    "Ritningsnummer",
    "Ritningsnummer Projekt",
    "Skala",
    "Skapad Av",
    "Stationsnamn",
    "Teknikområde",
    "Title",
    "Uppdragsnummer",
    "Konstruktionsnummer",
    "Ändring",
    "Översiktsplan",
]

# --------------------------------------------------
# OCR initialisation (Swedish + English)
# --------------------------------------------------
reader = easyocr.Reader(["sv", "en"], gpu=False)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def list_images(image_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = " ".join(text.split())
    return text


def ocr_lines_from_crop(crop: Image.Image, max_lines: int = 3) -> List[str]:
    """
    Run EasyOCR on the crop and return up to max_lines lines.
    """
    np_img = np.array(crop)
    results = reader.readtext(np_img, detail=0)  # list of strings

    lines: List[str] = []
    for t in results:
        for line in str(t).splitlines():
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= max_lines:
                break
        if len(lines) >= max_lines:
            break

    return lines[:max_lines]


def clean_andr_field(tokens: List[str]) -> str:
    """
    Clean the 'Ändring' field based on OCR tokens.
    A simplified but robust version capturing your typical formats.
    """
    import re

    raw = "".join(t.strip() for t in tokens if t and t.strip())
    raw = raw.strip()
    raw_up = raw.upper()

    # Field is just 'ÄNDR.' etc -> treat as '-'
    if raw_up.replace(" ", "") in ("ÄNDR.", "ÄNDR"):
        return "-"

    underscore_chars = set(["_", "-", "—", "–", "|", "L", "I", ".", "‒"])

    if raw_up == "" or all(ch in underscore_chars for ch in raw_up):
        return "_"

    # "__2" -> "_.2"
    m = re.match(r"^[_\-\—\–\.]+(\d+)$", raw_up)
    if m:
        return f"_.{m.group(1)}"

    if raw_up in underscore_chars:
        return "_"

    # "__B" -> "_B"
    m = re.match(r"^[_\-\—\–\.]+([A-ZÅÄÖ])$", raw_up)
    if m:
        return f"_{m.group(1)}"

    # "A2" or "A.2" or "A..2" -> "A.2"
    m = re.match(r"^([A-ZÅÄÖ])\.*(\d+)$", raw_up)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # Default: return original (normalised spacing)
    return raw_up


# --------------------------------------------------
# Core: process a single image with YOLO + OCR
# --------------------------------------------------
def process_single_image(
    pil_img: Image.Image,
    fname: str,
    model: YOLO,
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """
    Run YOLO on a single image, OCR each detected label region,
    and return a dict with keys: "Image" + LABELS.
    """
    row: Dict[str, Any] = {"Image": fname}
    for label in LABELS:
        row[label] = ""

    # YOLO inference
    np_img = np.array(pil_img)
    results = model.predict(source=np_img, verbose=False)

    if not results:
        return row

    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        return row

    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), cls_id in zip(boxes, classes):
        label_name = str(class_names[int(cls_id)])

        # We assume your YOLO class names match the labels exactly
        if label_name not in LABELS:
            continue

        # Crop region
        crop = pil_img.crop((x1, y1, x2, y2))

        # Special handling for Ändring
        if label_name == "Ändring":
            tokens = reader.readtext(np.array(crop), detail=0)
            cleaned = clean_andr_field(tokens)
            row["Ändring"] = cleaned
        else:
            lines = ocr_lines_from_crop(crop, max_lines=3)
            text = "\n".join(lines)
            text = normalize_text(text)

            # If multiple boxes for same label, append
            if row[label_name]:
                if text:
                    row[label_name] = f"{row[label_name]}\n{text}"
            else:
                row[label_name] = text

    return row


# --------------------------------------------------
# 1) Folder-based extraction (for local scripts, optional)
# --------------------------------------------------
def run_extraction(
    image_dir: str | Path,
    excel_out: str | Path,
    model_path: str | Path,
) -> pd.DataFrame:
    """
    Run YOLO + OCR on all images in image_dir and save rawData.xlsx.
    """
    image_dir = Path(image_dir)
    excel_out = Path(excel_out)

    model = YOLO(str(model_path))
    class_names = model.names  # {class_id: class_name}

    rows: List[Dict[str, Any]] = []

    for img_path in list_images(image_dir):
        pil_img = Image.open(img_path).convert("RGB")
        row = process_single_image(pil_img, img_path.name, model, class_names)
        rows.append(row)

    all_columns = ["Image"] + LABELS
    df = pd.DataFrame(rows, columns=all_columns)
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_out, index=False)

    return df


# --------------------------------------------------
# 2) Page-based extraction (used by Streamlit)
# --------------------------------------------------
def extract_from_pages(
    pages: List[Tuple[Image.Image | np.ndarray, str]],
    model_path: str | Path,
) -> pd.DataFrame:
    """
    For Streamlit:
    pages: list of (img, filename) tuples
           img can be a NumPy array OR a PIL.Image.Image
    model_path: path to YOLO weights (e.g. weights/best.pt)

    Returns: DataFrame with columns ["Image"] + LABELS.
    """
    model = YOLO(str(model_path))
    class_names = model.names

    rows: List[Dict[str, Any]] = []

    for img, fname in pages:
        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            pil_img = Image.fromarray(img)

        row = process_single_image(pil_img, fname, model, class_names)
        rows.append(row)

    all_columns = ["Image"] + LABELS
    df = pd.DataFrame(rows, columns=all_columns)
    return df


# --------------------------------------------------
# CLI example (optional)
# --------------------------------------------------
if __name__ == "__main__":
    # Example: run on all images in ./cropsImages_flat with ./weights/best.pt
    default_image_dir = Path("cropsImages_flat")
    default_model = Path("weights") / "best.pt"
    default_excel_out = Path("rawData.xlsx")

    if not default_model.exists():
        print(f"❌ Model file not found: {default_model}")
    elif not default_image_dir.exists():
        print(f"❌ Image folder not found: {default_image_dir}")
    else:
        df_result = run_extraction(
            image_dir=default_image_dir,
            excel_out=default_excel_out,
            model_path=default_model,
        )
        print("✅ Extraction completed:", default_excel_out)
        print(df_result.head())
