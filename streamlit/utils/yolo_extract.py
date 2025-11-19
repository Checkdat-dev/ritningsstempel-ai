import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import easyocr
import re

# ===================== CONFIG =====================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "best.pt")
MODEL_PATH = os.path.normpath(MODEL_PATH)

INPUT_FOLDER = r"D:\Checkdat_LIA\Metadata_extraction\dataset\unlabelled"
OUTPUT_EXCEL = "extract_all_raw.xlsx"

SAVE_DEBUG_CROPS = True
DEBUG_DIR = "debug_crops"

# ===================== LOAD YOLO & OCR =====================

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["sv", "en"], gpu=False)

# ===================== FIELD DEFINITIONS =====================

ORDER = [
    "ANDR", "ANLAGGNINGSTYP", "AVDELNING", "BANDEL", "BLAD",
    "Beskrivning_1", "Beskrivning_2", "Beskrivning_3", "Beskrivning_4",
    "DATUM", "FORMAT", "GODKAND_AV", "GRANSKAD_AV", "GRANSKNINGSSTATUS_SYFTE",
    "HANDLINGSTYP", "KILOMETER_METER", "KONSTRUKTIONSNUMMER", "LEVERANS_ANDRINGSPM",
    "LEVERANTOR_1", "LEVERANTOR_2", "NASTA_BLAD", "RITNINGSNUMMER_FORVALTNING",
    "RITNINGSNUMMER_PROJEKT", "SKALA", "SKAPAD_AV", "TEKNIKOMRADE",
    "TITEL", "UPPDRAGNUMMER"
]

# ===================== OCR HELPERS =====================

def extract_text_from_crop(image: np.ndarray) -> str:
    result = reader.readtext(image, detail=0, paragraph=True)
    return " ".join(result).strip() if result else ""

def segment_lines(crop: np.ndarray) -> list:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,
                               15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (crop.shape[1] // 2, 1))
    detect = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    ys = np.where(np.sum(detect, axis=1) > 0)[0]

    if len(ys) < 2:
        return [crop]

    cuts = np.where(np.diff(ys) > 10)[0]
    parts = []
    last = ys[0]
    for c in cuts:
        parts.append(crop[last:ys[c], :])
        last = ys[c + 1]
    parts.append(crop[last:ys[-1], :])

    return [p for p in parts if p.shape[0] > 8] or [crop]


# ===================== FINAL MASTER ANDR CLEAN-UP FUNCTION =====================

def clean_andr_field(tokens):
    raw = "".join(t.strip() for t in tokens if t.strip() != "")

    underscore_chars = set(["_", "-", "—", "–", "|", "l", "I", ".", "‒"])

    # EMPTY or all garbage → "_"
    if raw == "" or all(ch in underscore_chars for ch in raw):
        return "_"

    # _ + number
    m = re.match(r"^[_\-\—\–\.]+(\d+)$", raw)
    if m:
        return f"_.{m.group(1)}"

    # just underscore-like
    if raw in underscore_chars:
        return "_"

    # _ + letter
    m = re.match(r"^[_\-\—\–\.]+([A-ZÅÄÖ])$", raw)
    if m:
        return f"_{m.group(1)}"

    # A2 / A.2 / A..2
    m = re.match(r"^([A-ZÅÄÖ])\.?(\d+)$", raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    m = re.match(r"^([A-ZÅÄÖ])\.*(\d+)$", raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    # numbers only
    if raw.isdigit():
        return f"_.{raw}"

    # letter only
    if len(raw) == 1 and raw.isalpha():
        return raw.upper()

    return raw


# ===================== EXTRACTION =====================

def extract_raw_metadata(image: np.ndarray, filename: str) -> dict:

    img_pad = cv2.copyMakeBorder(image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255,255,255))
    results = model.predict(img_pad, conf=0.20, imgsz=960, augment=True)[0]
    extracted = {}

    print(f"\n📄 Processing: {filename}")

    if not results.boxes or len(results.boxes) == 0:
        for key in ORDER:
            extracted[key] = ""
        return extracted

    boxes_sorted = sorted(results.boxes, key=lambda b: b.xyxy[0][1])

    for box in boxes_sorted:
        cls = int(box.cls[0])
        label = results.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img_pad[y1:y2, x1:x2]

        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue

        if label == "Beskrivning_3":
            x1_adj = max(0, x1 - 185)
            crop = img_pad[y1:y2, x1_adj:x2]

        if SAVE_DEBUG_CROPS:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, f"{label}_{filename}"), crop)

        # OCR
        if label.startswith("Beskrivning"):
            parts = segment_lines(crop)
            raw_lines = [extract_text_from_crop(p) for p in parts]
            text = " ".join([t for t in raw_lines if t.strip()])
        else:
            text = extract_text_from_crop(crop)

        # ===================== ANDR FINAL MULTILINE FIX =====================
        if label == "ANDR":

            if "ANDR" in extracted and extracted["ANDR"] not in ["", None, "_"]:
                continue

            raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

            if len(raw_lines) >= 2:
                value_line = raw_lines[1]
            else:
                parts = segment_lines(crop)
                extracted_lines = [extract_text_from_crop(p) for p in parts if extract_text_from_crop(p)]

                if len(extracted_lines) >= 2:
                    value_line = extracted_lines[1]
                elif len(extracted_lines) == 1:
                    temp = extracted_lines[0]
                    temp = temp.replace("ÄNDR", "").replace("ANDR", "").replace(".", "").strip()
                    value_line = temp if temp else "_"
                else:
                    value_line = "_"

            value_line = value_line.replace(":", "_")

            tokens = (
                value_line.replace("|", "")
                        .replace("•", "")
                        .replace(":", "_")
                        .split()
            )

            clean_value = clean_andr_field(tokens)

            print(f"  🏷 {label}: {clean_value}")
            extracted[label] = clean_value
            continue
# =====================================================================

        print(f"  🏷 {label}: {text}")
        extracted[label] = text.strip()

    for key in ORDER:
        extracted.setdefault(key, "")

    return extracted

# ===================== WRAPPER FOR STREAMLIT APP =====================

def extract_metadata_from_stamp(image: np.ndarray) -> dict:
    """
    Wrapper for Streamlit app.
    Takes an image array and returns YOLO+OCR metadata.
    """
    return extract_raw_metadata(image, "uploaded_image")


# ===================== BULK PROCESS =====================

def process_folder():
    rows = []
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"\n🔎 Found {len(image_files)} images in folder: {INPUT_FOLDER}\n")

    for img_name in tqdm(image_files, desc="Extracting", unit="file"):
        full_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(full_path)
        metadata = extract_raw_metadata(img, img_name)
        metadata["filename"] = img_name
        rows.append(metadata)

    df = pd.DataFrame(rows, columns=["filename"] + ORDER)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n🎉 Extraction complete! Raw data saved as: {OUTPUT_EXCEL}")
