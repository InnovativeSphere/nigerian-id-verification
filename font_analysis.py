"""
font_analysis.py — Font consistency check for V2 fraud detection.
Compares visual font characteristics across different fields on the same
document. If any field deviates significantly from the card's own median
profile, it is flagged as a possible digital edit.
"""

import cv2
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# CONFIGURABLE THRESHOLDS (can be moved to config.py later)
# ------------------------------------------------------------
_STROKE_DEVIATION_THRESHOLD = 1.5   # factor above median stroke width to flag
_SHARPNESS_DEVIATION_THRESHOLD = 2.0  # factor above median sharpness variance to flag
_SPACING_DEVIATION_THRESHOLD = 1.8   # factor above median spacing regularity to flag
_MIN_FIELDS_REQUIRED = 3            # need at least this many fields to do a meaningful comparison


def _measure_stroke_width(text_region: np.ndarray) -> float:
    """
    Estimate the average stroke width of text in a binary region.
    Uses the distance transform to find the median thickness.
    """
    if text_region is None or text_region.size == 0:
        return 0.0
    # Ensure binary
    if len(text_region.shape) == 3:
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = text_region.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Distance transform: distance from each white pixel to nearest black pixel
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    # Median distance multiplied by 2 gives approximate stroke width
    stroke_width = np.median(dist[dist > 0]) * 2 if np.any(dist > 0) else 0.0
    return float(stroke_width)


def _measure_edge_sharpness(text_region: np.ndarray) -> float:
    """
    Measure edge sharpness using Laplacian variance.
    Higher variance = sharper edges / more detail.
    """
    if text_region is None or text_region.size == 0:
        return 0.0
    if len(text_region.shape) == 3:
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = text_region.copy()
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _measure_character_spacing(text_region: np.ndarray) -> float:
    """
    Measure the regularity of horizontal gaps between characters.
    Returns the standard deviation of gap widths — lower = more regular.
    """
    if text_region is None or text_region.size == 0:
        return 0.0
    if len(text_region.shape) == 3:
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = text_region.copy()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours (individual characters)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return 0.0

    # Get bounding boxes and sort left‑to‑right
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b: b[0])  # sort by x coordinate

    # Calculate horizontal gaps between consecutive characters
    gaps = []
    for i in range(len(boxes) - 1):
        x1, _, w1, _ = boxes[i]
        x2, _, _, _ = boxes[i + 1]
        gap = x2 - (x1 + w1)
        if gap >= 0:
            gaps.append(gap)

    if not gaps:
        return 0.0

    # Return standard deviation of gaps as a measure of irregularity
    return float(np.std(gaps))


def _compute_median_signatures(field_regions: dict, image: np.ndarray) -> dict:
    """
    Compute the median stroke width, sharpness, and spacing across all fields.
    Returns a dictionary with the median values.
    """
    signatures = {"stroke": [], "sharpness": [], "spacing": []}
    for field, bbox in field_regions.items():
        if bbox is None:
            continue
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        signatures["stroke"].append(_measure_stroke_width(crop))
        signatures["sharpness"].append(_measure_edge_sharpness(crop))
        signatures["spacing"].append(_measure_character_spacing(crop))

    medians = {}
    for key, values in signatures.items():
        if values:
            medians[key] = float(np.median(values))
        else:
            medians[key] = 0.0
    return medians


def analyze_font_consistency(image: np.ndarray, field_regions: dict) -> dict:
    """
    Compare font characteristics across fields on the same document.
    Any field that deviates significantly from the card's own median
    is flagged as a possible digital tamper.

    Args:
        image: The straightened BGR document image.
        field_regions: dict mapping field names to (x,y,w,h) tuples.

    Returns:
        Dictionary with consistency assessment and flagged fields.
    """
    if image is None or not field_regions or len(field_regions) < _MIN_FIELDS_REQUIRED:
        return {
            "consistent": True,
            "flagged_fields": [],
            "font_signatures_detected": 0,
            "details": "Not enough fields to perform font analysis."
        }

    # Compute card‑level medians
    medians = _compute_median_signatures(field_regions, image)
    if medians["stroke"] == 0.0 and medians["sharpness"] == 0.0:
        return {
            "consistent": True,
            "flagged_fields": [],
            "font_signatures_detected": 0,
            "details": "Could not extract reliable font measurements from image."
        }

    flagged_fields = []
    # Compare each field against the card medians
    for field, bbox in field_regions.items():
        if bbox is None:
            continue
        x, y, w, h = bbox
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        stroke = _measure_stroke_width(crop)
        sharpness = _measure_edge_sharpness(crop)
        spacing = _measure_character_spacing(crop)

        deviations = 0
        if medians["stroke"] > 0 and (stroke > medians["stroke"] * _STROKE_DEVIATION_THRESHOLD or stroke < medians["stroke"] / _STROKE_DEVIATION_THRESHOLD):
            deviations += 1
        if medians["sharpness"] > 0 and (sharpness > medians["sharpness"] * _SHARPNESS_DEVIATION_THRESHOLD or sharpness < medians["sharpness"] / _SHARPNESS_DEVIATION_THRESHOLD):
            deviations += 1
        if medians["spacing"] > 0 and (spacing > medians["spacing"] * _SPACING_DEVIATION_THRESHOLD or spacing < medians["spacing"] / _SPACING_DEVIATION_THRESHOLD):
            deviations += 1

        if deviations >= 2:
            flagged_fields.append(field)

    consistent = len(flagged_fields) == 0
    details = ""
    if not consistent:
        details = (
            f"Fields {', '.join(flagged_fields)} show font characteristics "
            "(stroke width, edge sharpness, and/or character spacing) "
            "inconsistent with the rest of the document."
        )
    else:
        details = "All fields show consistent font characteristics."

    logger.info(
        f"Font analysis complete: consistent={consistent}, "
        f"flagged={flagged_fields if flagged_fields else 'none'}"
    )

    return {
        "consistent": consistent,
        "flagged_fields": flagged_fields,
        "font_signatures_detected": 1 if consistent else (1 + len(set(flagged_fields))),
        "details": details
    }