import os
import numpy as np
import re

ORDER = [
    "ANDR", "ANLAGGNINGSTYP", "AVDELNING", "BANDEL", "BLAD",
    "Beskrivning_1", "Beskrivning_2", "Beskrivning_3", "Beskrivning_4",
    "DATUM", "FORMAT", "GODKAND_AV", "GRANSKAD_AV", "GRANSKNINGSSTATUS_SYFTE",
    "HANDLINGSTYP", "KILOMETER_METER", "KONSTRUKTIONSNUMMER", "LEVERANS_ANDRINGSPM",
    "LEVERANTOR_1", "LEVERANTOR_2", "NASTA_BLAD", "RITNINGSNUMMER_FORVALTNING",
    "RITNINGSNUMMER_PROJEKT", "SKALA", "SKAPAD_AV", "TEKNIKOMRADE",
    "TITEL", "UPPDRAGNUMMER"
]

# ========= Lazy-load YOLO model =========
_yolo_model = None
_ocr_reader = None

def load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("best.pt")
    return _yolo_model

def load_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["sv", "en"], gpu=False)
    return _ocr_reader

# ========= Helpers =========

def extract_text_from_crop(image):
    reader = load_ocr()
    result = reader.readtext(image, detail=0, paragraph=True)
    return " ".join(result).strip() if result else ""

def segment_lines(crop):
    import cv2   # moved inside function
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (crop.shape[1] // 2, 1))
    detect = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

    ys = np.where(np.sum(detect, axis=1) > 0)[0]

    if len(ys) < 2:
        return [crop]

    cuts = np.where(np.diff(ys) > 10)[0]
    parts = []
    last = ys[0]
    for c in cuts:
        parts.append(crop[last:ys[c]])
        last = ys[c + 1]
    parts.append(crop[last:ys[-1]])

    return parts
def clean_andr_field(tokens):
    raw = "".join(t.strip() for t in tokens if t.strip() != "")
    underscore_chars = set(["_", "-", "—", "–", "|", "l", "I", ".", "‒"])

    if raw == "" or all(ch in underscore_chars for ch in raw):
        return "_"

    m = re.match(r"^[_\-\—\–\.]+(\d+)$", raw)
    if m:
        return f"_.{m.group(1)}"

    if raw in underscore_chars:
        return "_"

    m = re.match(r"^[_\-\—\–\.]+([A-ZÅÄÖ])$", raw)
    if m:
        return f"_{m.group(1)}"

    m = re.match(r"^([A-ZÅÄÖ])\.?(\d+)$", raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    m = re.match(r"^([A-ZÅÄÖ])\.*(\d+)$", raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"

    if raw.isdigit():
        return f"_{raw}"

    if len(raw) == 1 and raw.isalpha():
        return raw.upper()

    return raw

# ========= Main extraction =========

def extract_raw_metadata(image: np.ndarray, filename: str) -> dict:
    import cv2   # moved inside function

    model = load_yolo()

    img_pad = cv2.copyMakeBorder(
        image, 40, 40, 40, 40,
        cv2.BORDER_CONSTANT,
        value=(255,255,255)
    )

    results = model.predict(img_pad, conf=0.20, imgsz=960)[0]
    extracted = {}

    if not results.boxes:
        for key in ORDER:
            extracted[key] = ""
        return extracted

    boxes_sorted = sorted(results.boxes, key=lambda b: b.xyxy[0][1])

    for box in boxes_sorted:
        cls = int(box.cls[0])
        label = results.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_pad[y1:y2, x1:x2]

        if label.startswith("Beskrivning"):
            parts = segment_lines(crop)
            text = " ".join([extract_text_from_crop(p) for p in parts])
        else:
            text = extract_text_from_crop(crop)

        extracted[label] = text.strip()

    for key in ORDER:
        extracted.setdefault(key, "")

    return extracted

    if label == "ANDR":

            if "ANDR" in extracted and extracted["ANDR"] not in ["", None, "_"]:
                continue

            raw_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

            if len(raw_lines) >= 2:
                value_line = raw_lines[1]
            else:
                parts = segment_lines(crop)
                extracted_lines = [
                    extract_text_from_crop(p)
                    for p in parts if extract_text_from_crop(p)
                ]

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

        extracted[label] = text.strip()

    for key in ORDER:
        extracted.setdefault(key, "")

    return extracted

def extract_metadata_from_stamp(image: np.ndarray) -> dict:
    return extract_raw_metadata(image, "uploaded")
