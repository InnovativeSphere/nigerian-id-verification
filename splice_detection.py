"""
splice_detection.py — Photo overlay / splice detection for V2 fraud detection.
Analyzes noise texture, edge artifacts, and lighting direction around the
face region to detect digitally pasted or replaced photographs.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from logger import get_logger

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# CONFIGURABLE THRESHOLDS
# ----------------------------------------------------------------------
_NOISE_HIST_DISTANCE_THRESHOLD = 0.35   # Chi‑squared distance above this → flag
_EDGE_DENSITY_RATIO_THRESHOLD = 1.5      # boundary edge density vs background
_LIGHTING_ANGLE_DIFF_THRESHOLD = 40.0    # degrees difference to flag
_FACE_EXPAND_PX = 25                     # pixels to expand face region for background sampling


def _compute_lbp_histogram(region: np.ndarray) -> np.ndarray:
    """
    Compute a normalized LBP histogram for a grayscale image region.
    LBP captures local texture / noise patterns independent of overall brightness.
    """
    if region is None or region.size == 0:
        return np.zeros(256, dtype=np.float64)
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    # Uniform LBP with radius=1, 8 points
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    return hist.astype(np.float64)


def _histogram_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Chi‑squared distance between two histograms.
    0 = identical, higher = more different.
    """
    if hist1.sum() == 0 or hist2.sum() == 0:
        return 1.0
    # Avoid division by zero
    diff = hist1 - hist2
    denom = hist1 + hist2 + 1e-10
    return float(np.sum(diff ** 2 / denom) / 2.0)


def _boundary_edge_density(image: np.ndarray, face_bbox: tuple) -> float:
    """
    Compute edge density (Canny edges) in a thin strip along the face boundary.
    Returns the ratio of edge pixels to total pixels in that strip.
    """
    x, y, w, h = face_bbox
    # Define a boundary strip: a few pixels just inside and outside the face box
    # We'll take the perimeter region of the face bbox
    h_img, w_img = image.shape[:2]

    # Outer boundary (dilated by 3px)
    x1_out = max(0, x - 3)
    y1_out = max(0, y - 3)
    x2_out = min(w_img, x + w + 3)
    y2_out = min(h_img, y + h + 3)

    # Inner boundary (eroded by 3px)
    x1_in = max(0, x + 3)
    y1_in = max(0, y + 3)
    x2_in = min(w_img, x + w - 3)
    y2_in = min(h_img, y + h - 3)

    # The boundary strip is the difference
    if x1_out >= x2_out or y1_out >= y2_out:
        return 0.0

    # Extract strip as outer minus inner
    outer_region = image[y1_out:y2_out, x1_out:x2_out]
    # Create a mask for the strip
    strip_mask = np.zeros(outer_region.shape[:2], dtype=np.uint8)
    cv2.rectangle(strip_mask, (x1_out - x1_out, y1_out - y1_out),
                  (x2_out - x1_out - 1, y2_out - y1_out - 1), 255, -1)
    cv2.rectangle(strip_mask, (x1_in - x1_out, y1_in - y1_out),
                  (x2_in - x1_out - 1, y2_in - y1_out - 1), 0, -1)

    # Compute Canny edges on outer region
    gray = cv2.cvtColor(outer_region, cv2.COLOR_BGR2GRAY) if len(outer_region.shape) == 3 else outer_region.copy()
    edges = cv2.Canny(gray, 50, 150)
    edges_in_strip = cv2.bitwise_and(edges, edges, mask=strip_mask)
    edge_pixels = np.sum(edges_in_strip > 0)
    total_pixels = np.sum(strip_mask > 0)
    if total_pixels == 0:
        return 0.0
    return edge_pixels / total_pixels


def _estimate_lighting_direction(region: np.ndarray) -> float:
    """
    Estimate the dominant lighting direction (in degrees) from a grayscale region.
    Uses gradient orientation histogram.
    Returns angle in degrees (0 = from right, 90 = from top).
    """
    if region is None or region.size == 0:
        return 0.0
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    # Use only pixels with significant gradient magnitude
    threshold = np.percentile(magnitude, 70)
    mask = magnitude > threshold
    if not np.any(mask):
        return 0.0
    # Weighted average of angles (circular mean)
    angles = np.deg2rad(angle[mask])
    weights = magnitude[mask]
    mean_x = np.sum(weights * np.cos(angles))
    mean_y = np.sum(weights * np.sin(angles))
    avg_angle = np.arctan2(mean_y, mean_x) * 180 / np.pi
    return avg_angle % 360


def detect_splice(image: np.ndarray, face_bbox: tuple) -> dict:
    """
    Detect whether the face photograph on an ID has been digitally spliced in.

    Args:
        image: The straightened BGR document image.
        face_bbox: (x, y, w, h) bounding box of the face region.

    Returns:
        Dictionary with splice assessment and triggered signals.
    """
    if image is None or face_bbox is None:
        return {
            "splice_suspected": False,
            "signals_triggered": [],
            "signals_checked": 0,
            "details": "No face region provided for splice analysis."
        }

    x, y, w, h = face_bbox
    h_img, w_img = image.shape[:2]

    # Extract face region
    face_region = image[max(0,y):min(h_img,y+h), max(0,x):min(w_img,x+w)]
    if face_region.size == 0:
        return {
            "splice_suspected": False,
            "signals_triggered": [],
            "signals_checked": 0,
            "details": "Face region is empty."
        }

    # Extract background region: an expanded ring around the face
    x1_bg = max(0, x - _FACE_EXPAND_PX)
    y1_bg = max(0, y - _FACE_EXPAND_PX)
    x2_bg = min(w_img, x + w + _FACE_EXPAND_PX)
    y2_bg = min(h_img, y + h + _FACE_EXPAND_PX)
    bg_region = image[y1_bg:y2_bg, x1_bg:x2_bg]

    # If the face takes up most of the card, background sampling is unreliable
    if bg_region.size == 0 or (bg_region.shape[0] * bg_region.shape[1] < 100):
        return {
            "splice_suspected": False,
            "signals_triggered": [],
            "signals_checked": 0,
            "details": "Not enough background area to perform splice analysis."
        }

    signals = []

    # ---- Signal 1: Noise pattern mismatch (LBP) ----
    face_hist = _compute_lbp_histogram(face_region)
    bg_hist = _compute_lbp_histogram(bg_region)
    noise_distance = _histogram_distance(face_hist, bg_hist)
    noise_flag = noise_distance > _NOISE_HIST_DISTANCE_THRESHOLD
    if noise_flag:
        signals.append("noise_pattern_mismatch")

    # ---- Signal 2: Edge artifact at boundary ----
    edge_density = _boundary_edge_density(image, face_bbox)
    # Compare with general edge density in the background
    bg_gray = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY) if len(bg_region.shape) == 3 else bg_region.copy()
    bg_edges = cv2.Canny(bg_gray, 50, 150)
    bg_edge_density = np.sum(bg_edges > 0) / bg_edges.size if bg_edges.size > 0 else 0
    edge_flag = (bg_edge_density > 0 and
                 edge_density > bg_edge_density * _EDGE_DENSITY_RATIO_THRESHOLD)
    if edge_flag:
        signals.append("edge_artifact")

    # ---- Signal 3: Lighting direction consistency ----
    face_lighting = _estimate_lighting_direction(face_region)
    bg_lighting = _estimate_lighting_direction(bg_region)
    # Circular angle difference
    angle_diff = abs(face_lighting - bg_lighting) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    lighting_flag = angle_diff > _LIGHTING_ANGLE_DIFF_THRESHOLD
    if lighting_flag:
        signals.append("lighting_mismatch")

    splice_suspected = len(signals) >= 2

    details = ""
    if splice_suspected:
        details = (
            f"Photo splice suspected. Triggered signals: {', '.join(signals)}. "
        )
        if "noise_pattern_mismatch" in signals:
            details += f"Noise texture mismatch (distance={noise_distance:.3f}). "
        if "edge_artifact" in signals:
            details += f"Elevated edge density at face boundary ({edge_density:.3f} vs bg {bg_edge_density:.3f}). "
        if "lighting_mismatch" in signals:
            details += f"Lighting direction mismatch (face={face_lighting:.1f}°, bg={bg_lighting:.1f}°)."
    else:
        details = "No significant splice indicators detected."

    logger.info(
        f"Splice analysis complete: suspected={splice_suspected}, "
        f"signals={signals}, noise_dist={noise_distance:.3f}, "
        f"edge_density={edge_density:.3f}, lighting_diff={angle_diff:.1f}°"
    )

    return {
        "splice_suspected": splice_suspected,
        "signals_triggered": signals,
        "signals_checked": 3,
        "details": details
    }