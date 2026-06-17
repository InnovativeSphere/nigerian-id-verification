"""
utils.py — Shared helper functions for the Nigerian ID Verification System.
Provides folder management, preprocessing pipelines, image saving,
and result formatting. Reusable across all modules.
"""

import os
import cv2
import numpy as np


# ----------------------------------------------------------------------
# FOLDER MANAGEMENT
# ----------------------------------------------------------------------
def ensure_folder_exists(folder: str) -> None:
    """
    Create a folder (and any parent directories) if it doesn't already exist.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")


# ----------------------------------------------------------------------
# PREPROCESSING VARIANTS
# ----------------------------------------------------------------------
def apply_standard_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing pipeline — used on the first extraction attempt.
    Steps: grayscale → binary threshold → deskew.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Binary threshold — clean text, remove background noise
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Deskew — correct slight tilts
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary = cv2.warpAffine(
            binary, matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return binary


def apply_enhanced_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    Enhanced preprocessing pipeline — used only on the retry attempt.
    Steps: grayscale → CLAHE contrast enhancement → sharpen kernel →
           aggressive adaptive threshold → deskew.

    This is a genuinely different strategy, not the same thing twice.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # CLAHE – enhances local contrast, brings out faint text
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Sharpen kernel – makes edges crisper for OCR
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # Aggressive adaptive threshold – local binarization
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 4
    )

    # Deskew – same as standard pipeline
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        binary = cv2.warpAffine(
            binary, matrix, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return binary

def save_face(face: np.ndarray, output_dir: str) -> str:
    """
    Save a cropped face image to the output directory with a timestamp filename.

    Args:
        face: BGR face image.
        output_dir: Folder where the image will be saved.

    Returns:
        The file path of the saved image.
    """
    from datetime import datetime
    ensure_folder_exists(output_dir)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"face_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, face)
    return filepath