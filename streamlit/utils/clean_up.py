import re
import pandas as pd

# ================================================================
#  LABEL PREFIXES TO REMOVE (Swedish + OCR variants)
# ================================================================
LABEL_WORDS = [
    "ANDR", "ÄNDR",
    "ANLÄGGNINGSTYP", "ANLAGGNINGSTYP",
    "AVDELNING",
    "BANDEL",
    "BLAD",
    "BESKRIVNING",
    "DATUM",
    "FORMAT",
    "GODKAND", "GODKÄND", "GODK'",
    "GRANSKAD",
    "GRANSKNINGSSTATUS",
    "SYFTE",
    "HANDLINGSTYP",
    "KILOMETER", "KILOME TER",
    "METER", "+METER",
    "AV",
    "KONSTRUKTIONSNUMMER",
    "LEVERANS",
    "LEVERANTOR", "LEVERANTÖR","LEVERANTDR","EVERANTÖR","LEVERANTDR",
    "NÄSTA", "NASTA",
    "RITNINGSNUMMER",
    "FÖRVALTNING", "FORVALTNING",
    "PROJEKT",
    "SKALA",
    "SKAPAD",
    "TEKNIKOMRADE", "TEKNIKOMRÅDE", "TEKNIKDMRADE","TEKNIKOMRÄDE",
    "TITEL",
    "UPPDRAGSNUMMER"
]


def remove_label_prefix(text: str):
    if not isinstance(text, str):
        return text

    t = text.strip()
    for word in LABEL_WORDS:
        pattern = rf"^{re.escape(word)}\s*"
        t = re.sub(pattern, "", t, flags=re.IGNORECASE)
    return t.strip()



def clean_nasta_blad(val):
    if pd.isna(val):
        return ""
    text = re.sub(r"\D", "", str(val).strip())
    if len(text) > 4 and text.startswith("1"):
        return text[1:]
    return text



def clean_skala(val):
    if pd.isna(val):
        return ""

    text = str(val).strip()
    text = text.replace(";", "/").replace(",", "/").replace(" ", "/")
    text = text.replace("\\", "/").replace("-", "/")

    parts = [p for p in text.split("/") if p.strip()]
    cleaned = []

    for p in parts:
        p = p.strip()

        if "1:" in p:
            cleaned.append(p[p.find("1:"):])
            continue

        m_dec = re.match(r"^1[\.\-](\d{2,4})$", p)
        if m_dec:
            cleaned.append(f"1:{m_dec.group(1)}")
            continue

        if p.isdigit():
            if len(p) >= 3:
                cleaned.append(f"1:{p[1:]}")
            else:
                cleaned.append(f"1:{p}")
            continue

    cleaned = list(dict.fromkeys(cleaned))
    return "/".join(cleaned)



def clean_granskningsstatus_syfte(val):
    if pd.isna(val):
        return ""

    text = str(val).lower().strip()

    for w in ["syfte", "skyfte", "skypte", "syfe"]:
        text = text.replace(w, "")

    if not text:
        return ""

    if any(p in text for p in ["godk", "g0dk", "gödk", "godkand", "godka"]):
        return "GODKÄND"

    if "gransk" in text or "grans" in text or "för" in text or "for" in text:
        return "FÖR GRANSKNING"

    return ""



def clean_konstrukt_and_andrings(val):
    if pd.isna(val):
        return ""

    text = str(val).upper().strip()

    remove_words = [
        "KONSTRUKTIONSNUMMER", "KONSTRUK", "KONSTRUKTION",
        "KDNSTRUK", "TIONSNUMMER", "NUMMER",
        "ANDRINGS-PM", "ÄNDRINGS-PM", "IÄNDRINGS-PM", "ZANDRINGS-PM",
        "ANDRINGSPM", "ÄNDRINGSPM",
        "LEVERANS", "LEVERANS/", "LEVERANS /","KONS TRUK","TIDNS"
    ]
    for w in remove_words:
        text = text.replace(w, "")

    text = text.replace("/", " ").replace(",", " ").replace("_", " ")
    return " ".join(text.split()).strip()



def clean_datum(val):
    if pd.isna(val):
        return ""

    t = str(val).strip()
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    m2 = re.search(r"(\d{5})-(\d{2})-(\d{2})", t)
    if m2 and m2.group(1).startswith("1"):
        return f"{m2.group(1)[1:]}-{m2.group(2)}-{m2.group(3)}"

    return ""



def clean_beskrivning_4(val):
    if pd.isna(val):
        return ""
    t = re.sub(r"[^A-ZÅÄÖ\s]", " ", str(val).upper().strip())
    t = re.sub(r"\s+", " ", t)

    det_variants = [
        "DETALJRITNING","JETALJRITNING","TALJRITNING",
        "ETALJRITNING","DE ALJRILIVIIVU",
        "DETALJRITNINGN","DETALJRITNIN",
    ]
    for dv in det_variants:
        if dv in t:
            return "DETALJRITNING"
    return ""



def digits_only(val):
    if pd.isna(val):
        return ""
    s = re.sub(r"\D", "", str(val))
    return s if s else ""



def clean_leverantor2(val):
    if pd.isna(val):
        return ""
    t = re.sub(r"^[0-9]+\s*", "", str(val).strip())
    return " ".join(t.split())



def clean_leverantor1(val):
    if pd.isna(val):
        return ""
    t = re.sub(r"^[0-9]+\s*", "", str(val).strip())
    return " ".join(t.split())



def clean_name_field(val):
    if pd.isna(val):
        return ""
    text = str(val).strip()

    text = re.sub(r"^[Aa][Vv]\s*", "", text)
    text = re.sub(r"\s*/\s*", "/", text)

    if "/" not in text:
        parts = text.split()
        if len(parts) >= 2:
            company = parts[0]
            name_part = " ".join(parts[1:])
        else:
            return text
    else:
        company, name_part = text.split("/", 1)
        company = company.strip()
        name_part = name_part.strip()

    name_part = name_part.replace("  ", " ").strip()

    m_already = re.match(r"^[A-ZÅÄÖ]\.[A-ZÅÄÖ]{3,}.*", name_part, flags=re.IGNORECASE)
    if m_already:
        clean_name = name_part
    else:
        tokens = name_part.split()
        if len(tokens) == 1:
            token = tokens[0]
            if "." in token:
                clean_name = token
            else:
                if len(token) >= 4:
                    clean_name = f"{token[0]}.{token[1:]}"
                else:
                    clean_name = token
        else:
            first = tokens[0]
            rest = "".join(tokens[1:])
            if len(rest) >= 3:
                clean_name = f"{first[0]}.{rest}"
            else:
                clean_name = " ".join(tokens)

    return f"{company}/{clean_name}"



def clean_beskrivning_3(val):
    if pd.isna(val):
        return ""
    if str(val).strip().lower() in ["nan", "none", "null", ""]:
        return ""
    return str(val).strip()



def clean_beskrivning_1(val):
    if pd.isna(val):
        return ""
    t = re.sub(r"[^A-ZÅÄÖ0-9\s]", " ", str(val).upper().strip())
    t = re.sub(r"\s+", " ", t).strip()

    t = (t.replace("05UU","0500")
           .replace("05UO","0500")
           .replace("O500","0500")
           .replace("MÄTVA","")
           .replace("DVAI","")
           .replace("DVA I","")
           .replace("MÄTVNI","")
           .replace("MÄTVN I",""))

    if re.search(r"\b0500\b", t):
        return "0500 STATION HAGA"
    if re.search(r"\b0700\b", t):
        return "0700 STATION KORSVÄGEN"
    if "CENTRALEN" in t:
        return "E02 CENTRALEN"

    good_words = ["STATION","CENTRALEN","KORSVÄGEN","HAGA"]
    if any(w in t for w in good_words):
        return t

    return ""



# --------------------------------------------------------------
# KM FIX HELPERS
# --------------------------------------------------------------
def fix_segment(seg):
    if not seg:
        return ""

    s = seg

    m = re.match(r"^(\d{3})(\d{2,4})$", s)
    if m:
        return f"{m.group(1)}+{m.group(2)}"

    if s.isdigit() and len(s) > 4:
        return s

    m2 = re.match(r"^(\d+/\d+)(\d+/\d+)$", s)
    if m2:
        return f"{m2.group(1)} - {m2.group(2)}"

    if "/" in s:
        return s

    if s.count("+") >= 2:
        return s

    m3 = re.match(r"^(\d{3})(\d+\.\d+)$", s)
    if m3:
        return f"{m3.group(1)}+{m3.group(2)}"

    return s.strip()

def fix_next_blad(value):
    if value is None:
        return ""
    # Remove spaces or non-digits
    digits = ''.join([c for c in value if c.isdigit()])
    if digits == "":
        return ""
    # Always pad to 4 digits
    return digits.zfill(4)


def clean_kilometer_meter(val):
    if pd.isna(val):
        return ""

    t = str(val).upper()
    t = t.replace(" ", "").replace(",", ".")
    t = re.sub(r"[A-Z~]*", "", t).replace("..", ".")

    # NEW RULE 1
    m_join = re.match(r"^(\d{3}\+\d+(?:\.\d+)?)(\d{3}\+\d+(?:\.\d+)?)$", t)
    if m_join:
        left = fix_segment(m_join.group(1))
        right = fix_segment(m_join.group(2))
        return f"{left} - {right}"

    # NEW RULE 2
    m_missing_plus = re.match(r"^(\d{3})(\d{2,4})(\d{3}\+\d+)$", t)
    if m_missing_plus:
        A = m_missing_plus.group(1)
        B = m_missing_plus.group(2)
        right = m_missing_plus.group(3)
        left = f"{A}+{B}"
        return f"{left} - {right}"

    # Case 1
    if "-" in t:
        left, right = t.split("-", 1)
        left = fix_segment(left)
        right = fix_segment(right)
        if left and right:
            return f"{left} - {right}"
        if left:
            return left
        if right:
            return right
        return ""

    # Case 2
    parts = t.split("+")
    if len(parts) == 4:
        A, B, C, D = parts
        return f"{A}+{B} - {C}+{D}"

    # Case 3
    return fix_segment(t)



def clean_granskad_av(val):
    if pd.isna(val):
        return ""

    t = str(val).strip()
    t = t.replace("[","").replace("]","")

    COMPANIES = ["TYRENS","TYRÉNS","SWECO","BERGAB","AMBERG"]

    m = re.match(r"^I\s*([A-Za-zÅÄÖÉ]+.*)$", t)
    if m:
        rest = m.group(1).strip()
        for c in COMPANIES:
            if rest.upper().startswith(c.replace("É","E")):
                return rest
    return t



# ================================================================
#  BUSINESS RULES
# ================================================================
def enforce_rules(df: pd.DataFrame) -> pd.DataFrame:
    df["AVDELNING"] = "TYRÉNS"
    df["UPPDRAGNUMMER"] = "TRV 2020/37994"
    df["GODKAND_AV"] = "TYRÉNS / JEB"
    df["GRANSKNINGSSTATUS_SYFTE"] = df["GRANSKNINGSSTATUS_SYFTE"].apply(clean_granskningsstatus_syfte)
    df["HANDLINGSTYP"] = "BYGGHANDLING"
    df["BANDEL"] = "604"
    df["FORMAT"] = "A1"
    df["TITEL"] = "VÄSTLÄNKEN"
    df["Beskrivning_4"] = df["Beskrivning_4"].apply(clean_beskrivning_4)

    df["RITNINGSNUMMER_PROJEKT"] = df["filename"].apply(
        lambda x: str(x).replace("_stamp.png","").replace("_stamp.PNG","")
    )

    def extract_last4(x):
        nums = re.findall(r"\d{4}", str(x))
        return nums[-1] if nums else ""

    df["BLAD"] = df["RITNINGSNUMMER_PROJEKT"].apply(extract_last4)
    
    df["NASTA_BLAD"] = df["NASTA_BLAD"].apply(clean_nasta_blad)
    return df



# ================================================================
#  MAIN CLEAN FUNCTION (Streamlit will call this)
# ================================================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    for col in df.columns:
        if col not in ["filename", "ANDR"]:
            df[col] = df[col].astype(str).apply(remove_label_prefix)

    if "DATUM" in df.columns:
        df["DATUM"] = df["DATUM"].apply(clean_datum)

    if "KONSTRUKTIONSNUMMER" in df.columns:
        df["KONSTRUKTIONSNUMMER"] = df["KONSTRUKTIONSNUMMER"].apply(clean_konstrukt_and_andrings)

    if "LEVERANS_ANDRINGSPM" in df.columns:
        df["LEVERANS_ANDRINGSPM"] = df["LEVERANS_ANDRINGSPM"].apply(clean_konstrukt_and_andrings)

        def keep_urb(val):
            if pd.isna(val):
                return ""
            t = str(val).upper()
            if "URB100211" in t:
                return "URB100211"
            if "JURB100211" in t:
                return "JURB100211"
            return ""

        df["LEVERANS_ANDRINGSPM"] = df["LEVERANS_ANDRINGSPM"].apply(keep_urb)

    if "LEVERANTOR_1" in df.columns:
        df["LEVERANTOR_1"] = df["LEVERANTOR_1"].apply(clean_leverantor1)

    if "LEVERANTOR_2" in df.columns:
        df["LEVERANTOR_2"] = df["LEVERANTOR_2"].apply(clean_leverantor2)

    if "NASTA_BLAD" in df.columns:
        df["NASTA_BLAD"] = df["NASTA_BLAD"].apply(digits_only)

    if "RITNINGSNUMMER_FORVALTNING" in df.columns:
        df["RITNINGSNUMMER_FORVALTNING"] = df["RITNINGSNUMMER_FORVALTNING"].apply(digits_only)

    if "SKAPAD_AV" in df.columns:
        df["SKAPAD_AV"] = df["SKAPAD_AV"].apply(clean_name_field)

    if "GRANSKAD_AV" in df.columns:
        df["GRANSKAD_AV"] = df["GRANSKAD_AV"].apply(clean_name_field)
        df["GRANSKAD_AV"] = df["GRANSKAD_AV"].apply(clean_granskad_av)

    if "SKALA" in df.columns:
        df["SKALA"] = df["SKALA"].apply(clean_skala)
    df["NASTA_BLAD"] = df["NASTA_BLAD"].apply(fix_next_blad)


    if "TEKNIKOMRADE" in df.columns:
        def clean_teknik(val):
            if pd.isna(val):
                return ""
            t = str(val)
            if "ANDRINGS-PM" in t or "ÄNDRINGS-PM" in t:
                return ""
            return t.strip()
        df["TEKNIKOMRADE"] = df["TEKNIKOMRADE"].apply(clean_teknik)

    if "KILOMETER_METER" in df.columns:
        df["KILOMETER_METER"] = df["KILOMETER_METER"].apply(clean_kilometer_meter)

    if "Beskrivning_3" in df.columns:
        df["Beskrivning_3"] = df["Beskrivning_3"].apply(clean_beskrivning_3)

    if "Beskrivning_1" in df.columns:
        df["Beskrivning_1"] = df["Beskrivning_1"].apply(clean_beskrivning_1)

    df = enforce_rules(df)
    return df
