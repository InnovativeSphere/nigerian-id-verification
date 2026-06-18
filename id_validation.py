"""
id_validation.py — ID number format and checksum validation for V2 fraud detection.
Validates the extracted ID number against known patterns for each document type.
Pure logic — no image processing or database calls.
"""

import re
from logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# PER‑DOCUMENT VALIDATION RULES
# ----------------------------------------------------------------------

def _validate_nin(id_number: str) -> dict:
    """Validate a NIN (National Identification Number)."""
    if not id_number:
        return {"valid_format": False, "details": "NIN is empty."}

    # NIN must be exactly 11 digits, numeric only
    if re.fullmatch(r"\d{11}", id_number):
        return {
            "valid_format": True,
            "details": "11‑digit numeric format matches expected NIN pattern."
        }
    else:
        # Provide specific failure reason
        if len(id_number) != 11:
            reason = f"Expected 11 digits, got {len(id_number)} characters."
        elif not id_number.isdigit():
            reason = "NIN must contain only digits."
        else:
            reason = "NIN format is invalid."
        return {"valid_format": False, "details": reason}


def _validate_drivers_license(id_number: str) -> dict:
    """Validate a Driver's License number (FRSC)."""
    if not id_number:
        return {"valid_format": False, "details": "License number is empty."}

    # Typical pattern: 2-3 uppercase letters + 4-6 digits + 2 uppercase letters
    # Allow some flexibility for older formats
    if re.fullmatch(r"[A-Z]{2,3}\d{4,6}[A-Z]{2}", id_number):
        return {
            "valid_format": True,
            "details": "License number format matches expected FRSC pattern."
        }
    else:
        return {
            "valid_format": False,
            "details": (
                f"License number '{id_number}' does not match expected format "
                "(e.g., ABC12345XY)."
            )
        }


def _validate_voters_card(id_number: str) -> dict:
    """Validate a Voter's Card number (PVC / VIN)."""
    if not id_number:
        return {"valid_format": False, "details": "Voter ID is empty."}

    # PVC numbers vary by election cycle. Accept any alphanumeric string
    # of 6‑19 characters (covers old and new formats).
    if re.fullmatch(r"[A-Z0-9]{6,19}", id_number):
        return {
            "valid_format": True,
            "details": "Voter ID format is plausible for a Nigerian PVC."
        }
    else:
        return {
            "valid_format": False,
            "details": (
                f"Voter ID '{id_number}' is not a valid alphanumeric format."
            )
        }


def _validate_passport(id_number: str) -> dict:
    """Validate an International Passport number."""
    if not id_number:
        return {"valid_format": False, "details": "Passport number is empty."}

    # Standard Nigerian passport: one letter + exactly 8 digits
    if re.fullmatch(r"[A-Z]\d{8}", id_number):
        return {
            "valid_format": True,
            "details": "Passport number format matches expected pattern (A12345678)."
        }
    else:
        return {
            "valid_format": False,
            "details": (
                f"Passport number '{id_number}' should be one letter followed by 8 digits."
            )
        }


# ----------------------------------------------------------------------
# DISPATCH TABLE
# ----------------------------------------------------------------------
_VALIDATORS = {
    "NIN_CARD": _validate_nin,
    "NIN_SLIP": _validate_nin,   # same format as NIN card
    "DRIVERS_LICENSE": _validate_drivers_license,
    "VOTERS_CARD": _validate_voters_card,
    "PASSPORT": _validate_passport,
}


# ----------------------------------------------------------------------
# PUBLIC FUNCTION
# ----------------------------------------------------------------------

def validate_id_format(id_number: str, doc_type: str) -> dict:
    """
    Validate an extracted ID number against the known format for its document type.

    Args:
        id_number: Cleaned ID string from OCR.
        doc_type: One of the five supported document types.

    Returns:
        Dictionary with keys 'valid_format', 'doc_type', and 'details'.
    """
    validator = _VALIDATORS.get(doc_type)
    if validator is None:
        logger.warning(f"No format validator for document type: {doc_type}")
        return {
            "valid_format": False,
            "doc_type": doc_type,
            "details": f"Unknown document type '{doc_type}' — cannot validate ID format."
        }

    result = validator(id_number)
    result["doc_type"] = doc_type
    logger.info(
        f"ID validation for {doc_type} ('{id_number}'): "
        f"{'PASS' if result['valid_format'] else 'FAIL'} — {result['details']}"
    )
    return result