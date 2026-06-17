"""
reader.py — Centralized EasyOCR wrapper for the ID Verification System.
"""

import easyocr
import numpy as np

_reader = easyocr.Reader(['en'])


def _clean_text(text: str) -> str:
    """Normalize OCR output: uppercase, keep alphanumeric and spaces."""
    return ''.join(c for c in text.upper() if c.isalnum() or c == ' ').strip()


def read_full(image: np.ndarray) -> str:
    if image is None or image.size == 0:
        return ""
    results = _reader.readtext(image)
    if not results:
        return ""
    results.sort(key=lambda r: r[0][0][0])
    raw = ' '.join(r[1].strip() for r in results)
    return _clean_text(raw)


def read_region(image: np.ndarray, x, y, w, h) -> str:
    if image is None or image.size == 0:
        return ""
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x+w), min(h_img, y+h)
    if x2 <= x1 or y2 <= y1:
        return ""
    region = image[y1:y2, x1:x2]
    results = _reader.readtext(region)
    if not results:
        return ""
    best = max(results, key=lambda r: r[2])
    return _clean_text(best[1])