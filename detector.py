"""
detector.py — Document detection for the Nigerian ID Verification System.
Uses morphological dilation to connect fragmented card edges,
then selects the largest rectangle by area for perspective correction.
"""

import cv2
import numpy as np
from typing import Optional


def detect_document(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. CLAHE contrast enhancement (brings out faint edges)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # 4. Canny edge detection (lower thresholds to capture more)
    edges = cv2.Canny(blurred, 20, 100)

    # 5. Dilate edges to connect broken fragments
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # 6. Find contours on the dilated edge image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 7. Select the largest contour (should be the card outline now)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ignore very small regions and huge ones (entire frame)
        if area < 500 or area > frame.shape[0] * frame.shape[1] * 0.9:
            continue
        card_cnt = cnt
        break

    if card_cnt is None:
        return None

    # 8. Get the minimum area bounding rectangle (rotated)
    rect = cv2.minAreaRect(card_cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # 9. Check aspect ratio (card‑like, not square)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
    aspect = rw / rh
    if not (1.2 <= aspect <= 2.2):
        return None

    # 10. Order corners for perspective transform
    pts = box.astype("float32")
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    ordered = np.array([tl, tr, br, bl], dtype="float32")

    # 11. Compute output size
    (tl, tr, br, bl) = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, matrix, (max_width, max_height))
    return warped