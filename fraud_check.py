"""
fraud_check.py — Fraud detection orchestrator for V2.
Runs all three checks (ID format, font consistency, splice detection),
combines their results into a single fraud assessment, calculates a
trust score, and determines the final fraud status.
"""

from id_validation import validate_id_format
from font_analysis import analyze_font_consistency
from splice_detection import detect_splice
from logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# CONFIGURABLE TRUST SCORE WEIGHTS
# ----------------------------------------------------------------------
_POINTS_PER_CLEAN_CHECK = 33      # maximum 99 if all three pass
_POINTS_FOR_FORMAT_VALID = 34     # slightly more for format (strong signal)
_POINTS_FOR_FONT_CLEAN = 33
_POINTS_FOR_SPLICE_CLEAN = 33
_PENALTY_FLAG = -15               # per flagged issue beyond the first


def _calculate_trust_score(
    format_result: dict,
    font_result: dict,
    splice_result: dict,
    issues: list
) -> int:
    """
    Calculate a 0‑100 trust score based on check results.
    Higher = more trustworthy.
    """
    score = 0

    # Award points for clean checks
    if format_result.get("valid_format", False):
        score += _POINTS_FOR_FORMAT_VALID
    if font_result.get("consistent", True):
        score += _POINTS_FOR_FONT_CLEAN
    if splice_result is not None and not splice_result.get("splice_suspected", False):
        score += _POINTS_FOR_SPLICE_CLEAN
    elif splice_result is None:
        # No face → splice check not run → treat as neutral
        score += _POINTS_FOR_SPLICE_CLEAN // 2

    # Penalize for each issue beyond the first
    for _ in range(len(issues)):
        score += _PENALTY_FLAG

    # Clamp to 0‑100
    return max(0, min(100, score))


def run_fraud_checks(
    image: 'np.ndarray',
    field_regions: dict,
    face_bbox: tuple | None,
    id_number: str | None,
    doc_type: str
) -> dict:
    """
    Run all three fraud checks and return a unified fraud assessment.

    Args:
        image: The straightened BGR document image.
        field_regions: dict mapping field names to (x,y,w,h) tuples.
        face_bbox: (x,y,w,h) of the detected face, or None.
        id_number: Extracted ID number string, or None.
        doc_type: One of the five supported document types.

    Returns:
        Full fraud_check dictionary with status, trust score, issues,
        and individual check results.
    """
    issues = []

    # ---- 1. ID Format Validation ----
    if id_number:
        format_result = validate_id_format(id_number, doc_type)
    else:
        format_result = {
            "valid_format": False,
            "doc_type": doc_type,
            "details": "No ID number provided for validation."
        }
    if not format_result.get("valid_format", False):
        issues.append("invalid_id_format")

    # ---- 2. Font Consistency Analysis ----
    if field_regions and len(field_regions) >= 3:
        font_result = analyze_font_consistency(image, field_regions)
    else:
        font_result = {
            "consistent": True,
            "flagged_fields": [],
            "font_signatures_detected": 0,
            "details": "Not enough field regions for font analysis."
        }
    if not font_result.get("consistent", True):
        issues.append("font_inconsistency")

    # ---- 3. Splice Detection ----
    if face_bbox is not None:
        splice_result = detect_splice(image, face_bbox)
    else:
        splice_result = {
            "splice_suspected": False,
            "signals_triggered": [],
            "signals_checked": 0,
            "details": "No face region provided. Splice analysis skipped."
        }
    if splice_result.get("splice_suspected", False):
        issues.append("photo_splice_suspected")

    # ---- Combine & Decide Status ----
    trust_score = _calculate_trust_score(format_result, font_result, splice_result, issues)

    if len(issues) == 0:
        status = "CLEAN"
    elif len(issues) == 1:
        status = "FLAGGED"
    else:
        status = "SUSPICIOUS"

    fraud_result = {
        "fraud_status": status,
        "trust_score": trust_score,
        "issues_detected": issues,
        "font_analysis": font_result,
        "splice_analysis": splice_result,
        "format_validation": format_result
    }

    logger.info(
        f"Fraud check complete: status={status}, trust={trust_score}, "
        f"issues={issues if issues else 'none'}"
    )

    return fraud_result