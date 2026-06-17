"""
extractor.py — Field extraction for the Nigerian ID Verification System.
Uses OCR keyword matching with upscaling for readability,
retry logic with enhanced preprocessing, and nationality defaulting.
"""

import re
import cv2
import numpy as np
from reader import read_full
from logger import get_logger

logger = get_logger(__name__)

# Patterns for critical fields
CRITICAL = {
    "surname": [r"SURNAME\s*[:]?\s*([\w]+)"],
    "first_name": [r"FIRST\s*NAME\s*[:]?\s*([\w]+)", r"GIVEN\s*NAMES?\s*[:]?\s*([\w]+)"],
    "id_number": [
        r"NIN\s*[:]?\s*(\d+)",
        r"L/NO\s*[:]?\s*([\w]+)",
        r"LNO\s*[:]?\s*([\w]+)",
        r"VIN\s*[:]?\s*([\w]+)",
        r"PASSPORT\s*NO\s*[:]?\s*([\w]+)",
    ],
}

# All possible optional fields
OPTIONAL = {
    "middle_name": [r"MIDDLE\s*NAME\s*[:]?\s*([\w]+)"],
    "date_of_birth": [r"DATE\s*OF\s*BIRTH\s*[:]?\s*([\w]+)", r"DOB\s*[:]?\s*([\w]+)"],
    "sex": [r"SEX\s*[:]?\s*([\w]+)"],
    "nationality": [r"NATIONALITY\s*[:]?\s*([\w]+)"],
    "height": [r"HEIGHT\s*[:]?\s*([\w.]+)", r"HT\s*[:]?\s*([\w.]+)"],
    "blood_group": [r"BLOOD\s*GROUP\s*[:]?\s*([\w+]+)", r"BG\s*[:]?\s*([\w+]+)"],
    "address": [r"ADDRESS\s*[:]?\s*(.+)"],
    "state": [r"STATE\s*[:]?\s*([\w]+)"],
    "issue_date": [],
    "expiry_date": [],
    "face_path": [],
    "confidence": [],
    "retried": [],
}


def _search_patterns(text: str, patterns: list) -> str | None:
    """Search a list of regex patterns in text and return the first capture."""
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_fields(image: np.ndarray, doc_type: str) -> dict:
    """Extract fields from a classified document using keyword matching."""
    if image is None or image.size == 0:
        return {**{k: None for k in (list(CRITICAL) + list(OPTIONAL))}}

    # Upscale for better OCR
    h, w = image.shape[:2]
    resized = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale if needed
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    full_text = read_full(gray)
    logger.info(f"OCR text ({len(full_text)} chars): {full_text[:300]}...")

    result = {}
    for field, patterns in CRITICAL.items():
        result[field] = _search_patterns(full_text, patterns)
    for field, patterns in OPTIONAL.items():
        result[field] = _search_patterns(full_text, patterns) if patterns else None

    # Default nationality for Nigerian IDs
    if not result.get("nationality"):
        match = re.search(r"(NGA|NIGERIA|NIGERIAN)", full_text, re.IGNORECASE)
        result["nationality"] = match.group(1).upper() if match else "NGA"

    return result


def extract_with_retry(image: np.ndarray, doc_type: str) -> dict:
    """Extract fields with retry logic using enhanced preprocessing."""
    # First attempt with standard preprocessing (grayscale only)
    result = extract_fields(image, doc_type)
    retried = False

    critical_missing = (
        not result.get("surname") or
        not result.get("first_name") or
        not result.get("id_number")
    )

    if critical_missing:
        logger.warning(
            f"Critical fields missing on first attempt for {doc_type}. Retrying."
        )
        # Second attempt with enhanced grayscale preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        result = extract_fields(sharpened, doc_type)
        retried = True

    final_missing = (
        not result.get("surname") or
        not result.get("first_name") or
        not result.get("id_number")
    )

    result["retried"] = retried
    if final_missing:
        result["confidence"] = "LOW"
        result["status"] = "PARTIAL"
    else:
        result["confidence"] = "HIGH"
        result["status"] = "SUCCESS"

    return result