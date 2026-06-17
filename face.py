"""
face.py — Face extraction for the Nigerian ID Verification System.
Uses Haar Cascade to detect the largest face on an ID document,
crops it, and saves it with a timestamp.
"""

import cv2
import numpy as np
from config import settings
from utils import save_face
from logger import get_logger

logger = get_logger(__name__)

# Load the Haar cascade once when the module is imported
_cascade = cv2.CascadeClassifier(settings.cascade_path)
if _cascade.empty():
    logger.warning("Haar cascade could not be loaded. Face extraction will be disabled.")


def extract_face(image: np.ndarray, doc_type: str) -> tuple[np.ndarray | None, str | None]:
    """
    Detect and extract the largest face from an ID document.
    Adds slight padding around the detected face for a better crop.
    """
    if _cascade.empty() or image is None or image.size == 0:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        logger.info(f"No face found on {doc_type}.")
        return None, None

    # Select the face with the largest area
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Add padding (20% of face size or 10px, whichever is larger)
    pad_x = max(int(w * 0.2), 10)
    pad_y = max(int(h * 0.2), 10)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    # Crop the face from the original colour image
    face_crop = image[y1:y2, x1:x2]

    # Save with timestamp
    saved_path = save_face(face_crop, settings.face_output_dir)
    logger.info(f"Face extracted from {doc_type} and saved to {saved_path}")
    return face_crop, saved_path