"""
classifier.py — Document type classification for the Nigerian ID Verification System.
Uses a two‑stage approach:
  1. Aspect ratio check → NIN Slip (width > 2.0× height)
  2. OCR keyword matching → NIN Card, Driver's License, Voter's Card, Passport
"""

import numpy as np
from reader import read_full

CARD_KEYWORDS = {
    "NIN_CARD": [
        # Official names and abbreviations
        "national identity card",
        "national identification card",
        "national identification number",
        "nigeria identification number",
        "federal republic of nigeria",
        "federal republic of nigeria identity card",
        "nimc",
        "nin",
        # Additional phrases often seen on the plastic card
        "identity card",
        "identification card",
        "national id",
        "nigeria id",
        "nimc card",
        "nin card",
        "national identity management commission",
    ],
    "DRIVERS_LICENSE": [
        "drivers licence",
        "driver's license",
        "drivers license",
        "driver licence",
        "frsc",
        "federal road safety corps",
        "national drivers license",
        "nigeria drivers license",
        "driving licence",
        "driving license",
        "motor vehicle licence",
        "motor vehicle license",
    ],
    "VOTERS_CARD": [
        "permanent voter",
        "permanent voters card",
        "permanent voter's card",
        "voter's card",
        "voters card",
        "pvc",
        "inec",
        "independent national electoral commission",
        "electoral commission",
        "voter registration",
        "voter identification",
        "polling unit",
    ],
    "PASSPORT": [
        "passport",
        "immigration",
        "republic of nigeria passport",
        "nigerian passport",
        "international passport",
        "ecowas passport",
        "travel document",
        "machine readable zone",
        "mrz",
        "surname",
        "given names",
        "nationality nigeria",
    ],
}


def _check_aspect_ratio(width: int, height: int) -> bool:
    """Return True if the document is wide enough to be a NIN Slip."""
    return width > height * 2.0  # NIN Slip is roughly 2.5:1


def classify_document(image: np.ndarray) -> str:
    """
    Determine the type of a Nigerian ID document.

    Args:
        image: Straightened, cropped BGR document image.

    Returns:
        One of: "NIN_SLIP", "NIN_CARD", "DRIVERS_LICENSE",
        "VOTERS_CARD", "PASSPORT", or "UNKNOWN".
    """
    if image is None or image.size == 0:
        return "UNKNOWN"

    h, w = image.shape[:2]

    # ---- Stage 1: Aspect Ratio Check ----
    if _check_aspect_ratio(w, h):
        return "NIN_SLIP"

    # ---- Stage 2: OCR Keyword Matching ----
    full_text = read_full(image)

    if not full_text:
        return "UNKNOWN"

    text_lower = full_text.lower()

    for doc_type, keywords in CARD_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type

    return "UNKNOWN"