# dataextractionsystem.py
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
import easyocr
from ultralytics import YOLO

# --------------------------------------------------
# 1) Global OCR reader (Swedish + English)
# --------------------------------------------------
reader = easyocr.Reader(["sv", "en"], gpu=False)


# --------------------------------------------------
# 2) Helpers
# --------------------------------------------------
def normalize_text(text: str) -> str:
    """Light whitespace normalisation."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_name(s: str) -> str:
    """
    Make a string safe for filenames:
    replace anything not A-Z, a-z, 0-9, ., _, - with underscore.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def clean_andr_field(tokens):
    """
    tokens: list of OCR token strings (e.g. from EasyOCR)
    """
    raw = "".join(t.strip() for t in tokens if t and t.strip() != "")
    raw = raw.strip()
    raw_up = raw.upper()

    # if field is 'ÄNDR.' (or 'ÄNDR'), map to '-'
    if raw_up.replace(" ", "") in ("ÄNDR.", "ÄNDR"):
        return "-"

    underscore_chars = set(["_", "-", "—", "–", "|", "L", "I", ".", "‒"])

    # Empty or only underscore-like chars → "_"
    if raw_up == "" or all(ch in underscore_chars for ch in raw_up):
        return "_"

    # underscores + number  e.g. "__2" -> "_.2"
    m = re.match(r"^[_\-\—\–\.]+(\d+)$", raw_up)
    if m:
        return f"_.{m.group(1)}"

    # just underscore-like
    if raw_up in underscore_chars:
        return "_"

    # underscores + letter e.g. "__B" -> "_B"
    m = re.match(r"^[_\-\—\–\.]+([A-ZÅÄÖ])$", raw_up)
    if m:
        return f"_{m.group(1)}"

    # A2 / A.2 / A..2 etc -> "A.2"
    m = re.match(r"^([A-ZÅÄÖ])\.?(\d+)$", raw_up)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    m = re.match(r"^([A-ZÅÄÖ])\.*(\d+)$", raw_up)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # numbers only -> "_.<num>"
    if raw_up.isdigit():
        return f"_.{raw_up}"

    # single letter only
    if len(raw_up) == 1 and raw_up.isalpha():
        return raw_up

    # fallback
    return raw_up


def detect_line_symbol_and_image(crop_img: Image.Image):
    gray = np.array(crop_img.convert("L"))
    h, w = gray.shape

    # Look mostly in the lower part of the box
    start = int(h * 0.5)
    end = int(h * 0.95)
    roi = gray[start:end, :]

    mask = roi < 230  # dark pixels
    if mask.mean() < 0.0001:
        return None, None

    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None, None

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    line_h = y_max - y_min + 1
    line_w = x_max - x_min + 1

    # wide & thin = line
    if line_h > 8:
        return None, None
    if line_w < w * 0.2:
        return None, None

    dark_vals = roi[mask]
    mean_dark = dark_vals.mean()

    if mean_dark < 70:
        symbol = "―"
    elif mean_dark < 150:
        symbol = "_"
    else:
        symbol = "-"

    abs_y1 = start + y_min
    abs_y2 = start + y_max
    abs_x1 = x_min
    abs_x2 = x_max

    symbol_img = crop_img.crop((abs_x1, abs_y1, abs_x2 + 1, abs_y2 + 1))

    return symbol, symbol_img


def ocr_first_three_lines(img: Image.Image, try_vertical_if_empty: bool = True):
    """
    Uses EasyOCR to extract up to the first 3 text 'lines' from an image.
    """

    def _lines_from_img(pil_img: Image.Image):
        arr = np.array(pil_img)
        # detail=1 -> [ [bbox, text, conf], ... ]
        results = reader.readtext(arr, detail=1)
        items = []

        for bbox, text, conf in results:
            text = text.strip()
            if not text:
                continue
            # bbox = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            y_coords = [p[1] for p in bbox]
            y_center = sum(y_coords) / len(y_coords)
            items.append((y_center, text))

        if not items:
            return []

        # sort by vertical center
        items.sort(key=lambda x: x[0])

        # group into "lines" based on y difference
        lines = []
        current_y = None
        current_tokens = []
        y_thresh = 10  # pixels

        for y, text in items:
            if current_y is None:
                current_y = y
                current_tokens = [text]
            elif abs(y - current_y) <= y_thresh:
                current_tokens.append(text)
            else:
                line = normalize_text(" ".join(current_tokens))
                if line:
                    lines.append(line)
                current_y = y
                current_tokens = [text]

        if current_tokens:
            line = normalize_text(" ".join(current_tokens))
            if line:
                lines.append(line)

        # Only first 3 lines
        return lines[:3]

    # 1) Try normal orientation (horizontal text)
    lines = _lines_from_img(img)
    if lines:
        return lines

    # 2) If nothing, try vertical text (rotate 90°)
    if try_vertical_if_empty:
        img_rot = img.rotate(90, expand=True)
        lines_rot = _lines_from_img(img_rot)
        return lines_rot

    return []


# --------------------------------------------------
# 3) Labels / columns
# --------------------------------------------------
LABELS = [
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


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return [p for p in folder.iterdir() if p.suffix.lower() in exts]


# --------------------------------------------------
# 4) Core: process a single image (PIL) into a row dict
# --------------------------------------------------
def process_single_image(
    pil_img: Image.Image,
    img_name: str,
    model,
    names,
    save_debug_crops: bool = False,
    debug_dir: Path | None = None,
    save_symbol_images: bool = False,
    symbol_dir: Path | None = None,
):
    w_img, h_img = pil_img.size

    # Prepare row
    row = {label: "" for label in LABELS}
    row["Image"] = img_name

    # Run YOLO
    results = model(pil_img)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return row

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        label_name = names.get(cls_id, str(cls_id))

        # Only care about the labels we defined
        if label_name not in LABELS:
            continue

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_img, x2)
        y2 = min(h_img, y2)

        crop = pil_img.crop((x1, y1, x2, y2))

        # Save debug crop (with safe filename)
        if save_debug_crops and debug_dir is not None:
            safe_label = safe_name(label_name)
            debug_name = f"{Path(img_name).stem}_{safe_label}_{i}.png"
            crop.save(debug_dir / debug_name)

        # Detect line symbol in crop
        symbol_char, symbol_img = detect_line_symbol_and_image(crop)
        if save_symbol_images and symbol_dir is not None and symbol_char is not None and symbol_img is not None:
            safe_label = safe_name(label_name)
            sym_name = f"{Path(img_name).stem}_{safe_label}_{i}_symbol.png"
            symbol_img.save(symbol_dir / sym_name)

        # -------------------------------
        # Special logic per field
        # -------------------------------
        if label_name == "Ändring":
            ocr_tokens = reader.readtext(np.array(crop), detail=0)  # list of strings
            cleaned_andr = clean_andr_field(ocr_tokens)

            # If only underscores / empty but we have a line symbol, use symbol
            if cleaned_andr in ["", "_"] and symbol_char is not None:
                row["Ändring"] = symbol_char
            else:
                row["Ändring"] = cleaned_andr

        else:
            # Default: take 1–3 lines (horizontal, fallback vertical)
            lines = ocr_first_three_lines(crop, try_vertical_if_empty=True)
            # Join lines with newline for Excel multi-line cell
            row[label_name] = "\n".join(lines)

    return row


# --------------------------------------------------
# 5) Folder-based extraction (optional CLI use)
# --------------------------------------------------
def run_extraction(
    image_dir,
    excel_out,
    model_path,
    save_debug_crops: bool = False,
    debug_dir: str | Path | None = None,
    save_symbol_images: bool = False,
    symbol_dir: str | Path | None = None,
):
    """
    Folder-based extraction (not used by Streamlit, but handy locally).
    """

    image_dir = Path(image_dir)
    excel_out = Path(excel_out)

    # Prepare optional dirs
    debug_dir = Path(debug_dir) if debug_dir is not None else image_dir / "debug_crops"
    symbol_dir = Path(symbol_dir) if symbol_dir is not None else image_dir / "symbol_crops"

    if save_debug_crops:
        os.makedirs(debug_dir, exist_ok=True)
    if save_symbol_images:
        os.makedirs(symbol_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO(str(model_path))
    names = model.names  # {class_id: class_name}

    data_rows = []
    image_paths = list_images(image_dir)

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        row = process_single_image(
            pil_img,
            img_path.name,
            model,
            names,
            save_debug_crops=save_debug_crops,
            debug_dir=debug_dir,
            save_symbol_images=save_symbol_images,
            symbol_dir=symbol_dir,
        )
        data_rows.append(row)

    # Save to Excel
    all_columns = ["Image"] + LABELS
    df = pd.DataFrame(data_rows, columns=all_columns)
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(excel_out, index=False)

    return str(excel_out)


# --------------------------------------------------
# 6) Page-based extraction for Streamlit (PDF -> images)
# --------------------------------------------------
def extract_from_pages(pages, model_path):
    """
    For Streamlit:
    pages: list of (img, filename) tuples
        - img can be a NumPy array or a PIL.Image.Image
    model_path: path to YOLO weights

    Returns: pd.DataFrame with columns ["Image"] + LABELS
    """
    # Load YOLO once
    model = YOLO(str(model_path))
    names = model.names

    data_rows = []

    for img, fname in pages:
        # Ensure we have a PIL image
        if isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        else:
            # assume NumPy array
            pil_img = Image.fromarray(img)

        row = process_single_image(
            pil_img,
            fname,
            model,
            names,
            save_debug_crops=False,
            debug_dir=None,
            save_symbol_images=False,
            symbol_dir=None,
        )
        data_rows.append(row)

    all_columns = ["Image"] + LABELS
    df = pd.DataFrame(data_rows, columns=all_columns)
    return df


# --------------------------------------------------
# 7) CLI example (optional)
# --------------------------------------------------
if __name__ == "__main__":
    default_model = "weights/best.pt"
    default_image_dir = "cropsImages_flat"
    default_excel_out = "rawData.xlsx"

    if Path(default_model).exists() and Path(default_image_dir).exists():
        run_extraction(
            image_dir=default_image_dir,
            excel_out=default_excel_out,
            model_path=default_model,
            save_debug_crops=True,
            save_symbol_images=True,
        )
        print("✅ Extraction completed:", default_excel_out)
    else:
        print("Set up weights/best.pt and cropsImages_flat/ to use CLI example.")
