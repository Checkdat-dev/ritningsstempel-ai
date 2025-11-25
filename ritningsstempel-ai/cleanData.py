import pandas as pd
import numpy as np   # still imported, but np.nan is no longer used
import unicodedata
import re

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

SWEDISH_FIXES = {
    "godkand": "godkänd",
    "GodkAND": "GODKÄND",
    "granskad": "granskad",
    "andring": "ändring",
    "oversiktsplan": "översiktsplan",
    "teknikomrade": "teknikområde",
    "nasta blad": "nästa blad",
    "leverantor": "leverantör",
}


def fix_swedish_ocr(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t_lower = text.lower()
    for wrong, right in SWEDISH_FIXES.items():
        if wrong in t_lower:
            pattern = re.compile(re.escape(wrong), flags=re.IGNORECASE)
            text = pattern.sub(right, text)
            t_lower = text.lower()
    return text


def normalize_swe(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def remove_label_from_column(df: pd.DataFrame, labels):
    """
    Remove the label word itself from each corresponding column,
    but DO NOT convert empty strings to NaN.
    """
    for label in labels:
        if label not in df.columns:
            continue

        orig = re.escape(label)
        norm = re.escape(normalize_swe(label))
        pattern = re.compile(rf"\b({orig}|{norm})\b", flags=re.IGNORECASE)

        df[label] = (
            df[label]
            .astype(str)
            .apply(lambda x: pattern.sub("", x))
            .str.strip()
        )
    return df


def clean_status(value):
    if pd.isna(value):
        return None
    s = str(value).upper()
    s = s.replace(" ", "")

    s = s.replace("FORGRANSKNING", "FÖRGRANSKNING")
    s = s.replace("FÖRGRANSK NING", "FÖRGRANSKNING")
    s = s.replace("GODKAND", "GODKÄND")
    s = s.replace("GODK ÄND", "GODKÄND")

    if "GODKÄND" in s:
        return "GODKÄND"
    if "FÖRGRANSKNING" in s:
        return "FÖR GRANSKNING"
    return None


def _apply_all_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core cleaning logic: takes a DataFrame and returns a cleaned DataFrame.
    Used by both clean_data(df) and run_cleaning(inp, out).
    """
    df = df.copy()

    # --- Anläggningstyp ---
    if "Anläggningstyp" in df.columns:
        df["Anläggningstyp"] = (
            df["Anläggningstyp"]
            .astype(str)
            .str.replace(r"\bANLAGGNINGSTYP\b\s*", "", case=False, regex=True)
            .str.strip()
        )

    # --- Format ---
    if "Format" in df.columns:
        variants = ["FDRMA T", "FORMA T"]
        pattern = "|".join(re.escape(v) for v in variants)
        col = "Format"
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Godkänd Av ---
    if "Godkänd Av" in df.columns:
        variants = [
            "ODKÄND AV",
            "GODKAND Av",
            "GODK'ÄND Av",
            "GODK'ÄND AV",
            "GODKÄND AV",
            "GODKÄND AV ",
            "GODKAND AV",
            "GODKAND AV ",
            "[",
        ]
        col = "Godkänd Av"
        pattern = "|".join(re.escape(v) for v in variants)
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Granskningsstatus/Syfte ---
    if "Granskningsstatus/Syfte" in df.columns:
        df["Granskningsstatus/Syfte"] = df["Granskningsstatus/Syfte"].apply(
            clean_status
        )

    # --- Kilometer & Meter ---
    if "Kilometer & Meter" in df.columns:
        col = "Kilometer & Meter"
        variants = [
            r"\bKILOMETER\s*[+&]\s*METER\b",
            r"\bKILOME\s*TER\+METER\b",
            r"\bKILODMETER\+METER\b",
        ]
        pattern = "|".join(variants)
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(pattern, "", regex=True, case=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Leverans/Ändrings-pm ---
    if "Leverans/Ändrings-pm" in df.columns:
        col = "Leverans/Ändrings-pm"
        variants = [
            "LEVERANS /ANDRINGS-PM",
            "LEVERANS / ÄNDRINGS-PM",
            "LEVERANS/ANDRINGS-PM",
            "LEVERANS/ÄNDRINGS-PM",
            "LEVERANS / ANDRINGS-PM",
            "LEVERANS /ÄNDRINGS-PM",
        ]
        pattern = "|".join(re.escape(v) for v in variants)
        df[col] = (
            df[col]
            .astype(str)
            .str.upper()
            .str.replace(pattern, "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Ändring ---
    if "Ändring" in df.columns:
        col = "Ändring"
        df[col] = (
            df[col]
            .astype(str)
            # Remove label variants like "ÄNDR.", "ÄNDR:", "ÄNDR", "ANDR.", "ANDR:"
            .str.replace(r"\bÄNDR[.:]?", "", regex=True, case=False)
            .str.replace(r"\bANDR[.:]?", "", regex=True, case=False)
            # In case label is stuck to following text: "ÄNDRA", "ANDRREV01"
            .str.replace(r"\bÄNDR(?=[A-ZÅÄÖ])", "", regex=True, case=False)
            .str.replace(r"\bANDR(?=[A-ZÅÄÖ])", "", regex=True, case=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Turn empty / literal "nan" strings into "-" (no np.nan used)
        df[col] = df[col].replace(["", "nan", "NaN"], "-")
        # Also replace real NaNs from Excel with "-"
        df[col] = df[col].fillna("-")

    # --- Title ---
    if "Title" in df.columns:
        df["Title"] = (
            df["Title"]
            .astype(str)
            .str.upper()
            .str.extract(r"(TRAFIKVERKET)", expand=False)
        )


        # NOTE: .str.extract() will still give NaN if "TYRENS" not found.
        # If you want to avoid *those* NaNs too, we can change this logic.

    # --- Remove label words from all label columns ---
    df = remove_label_from_column(df, LABELS)

    return df


# =============== PUBLIC API ===============

def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Used by Streamlit:
    - takes a raw DataFrame
    - returns a cleaned DataFrame
    """
    return _apply_all_cleaning(df_raw)


def run_cleaning(inp: str, out: str) -> str:
    """
    File-based version:
    - reads Excel from 'inp'
    - saves cleaned Excel to 'out'
    """
    df_raw = pd.read_excel(inp)
    df_clean = _apply_all_cleaning(df_raw)
    df_clean.to_excel(out, index=False)
    print("✅ Cleaned file saved to:", out)
    return out


if __name__ == "__main__":
    # manual test
    run_cleaning(inp="rawData.xlsx", out="cleanData.xlsx")
