"""
main.py — Orchestrator for the Nigerian ID Verification System.
Usage:
    python main.py --mode image --image test_images/nin_card_sample.jpg
    python main.py --mode live
"""

import argparse
import time
import json
import cv2
import numpy as np

from config import settings
from utils import ensure_folder_exists
from detector import detect_document
from classifier import classify_document
from extractor import extract_with_retry, extract_field_regions
from face import extract_face
from database import lookup_identity, insert_identity, update_fraud_record
from fraud_check import run_fraud_checks
from logger import get_logger

logger = get_logger(__name__)


def _process_document(image: np.ndarray) -> dict:
    """Run the full extraction pipeline with V2 fraud checks."""
    # 1. Detect and straighten the ID card
    doc = detect_document(image)
    if doc is None:
        return {"status": "ERROR", "message": "No document detected."}

    # 2. Classify the document type
    doc_type = classify_document(doc)
    if doc_type == "UNKNOWN":
        return {"status": "ERROR", "message": "Unable to classify document type."}

    # 3. Extract fields with retry
    extracted = extract_with_retry(doc, doc_type)

    # 4. Extract face (now returns bbox too)
    face_img, face_path, face_bbox = extract_face(doc, doc_type)
    extracted["face_path"] = face_path

    # 4b. Extract field regions for font analysis
    field_regions = extract_field_regions(doc, doc_type)

    # 4c. Run fraud checks (V2)
    fraud_result = run_fraud_checks(
        image=doc,
        field_regions=field_regions,
        face_bbox=face_bbox,
        id_number=extracted.get("id_number"),
        doc_type=doc_type
    )
    extracted["fraud_check"] = fraud_result

    # Ensure doc_type is passed to the database
    extracted["doc_type"] = doc_type

    # 5. Database lookup
    id_number = extracted.get("id_number")
    if id_number:
        existing = lookup_identity(id_number)
        if existing:
            extracted["db_status"] = "KNOWN"
            extracted["db_record"] = existing
        else:
            try:
                new_record = insert_identity(extracted)
                extracted["db_status"] = "NEW_ENTRY"
                extracted["db_record"] = new_record
            except Exception as e:
                logger.error(f"Failed to insert identity: {e}")
                extracted["db_status"] = "DB_ERROR"

        # Update fraud tracking columns if record exists
        if extracted["db_status"] in ("KNOWN", "NEW_ENTRY"):
            update_fraud_record(id_number, fraud_result)
    else:
        extracted["db_status"] = "NO_ID_NUMBER"

    # 6. Build final result
    result = {
        "doc_type": doc_type,
        "id_number": id_number,
        "surname": extracted.get("surname"),
        "first_name": extracted.get("first_name"),
        "middle_name": extracted.get("middle_name"),
        "date_of_birth": extracted.get("date_of_birth"),
        "sex": extracted.get("sex"),
        "nationality": extracted.get("nationality"),
        "face_path": face_path,
        "confidence": extracted.get("confidence", "UNKNOWN"),
        "retried": extracted.get("retried", False),
        "db_status": extracted.get("db_status"),
        "fraud_check": fraud_result,
        "flag_count": extracted.get("flag_count", 0)
    }

    return result


def run_image_mode(image_path: str) -> None:
    """Process a single static image."""
    logger.info(f"Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error(f"Could not read image from {image_path}")
        return

    result = _process_document(frame)
    print(json.dumps(result, indent=2, default=str))
    logger.info(f"Result: {result.get('status', 'SUCCESS')}")


def run_live_mode() -> None:
    """Continuously monitor the webcam feed."""
    logger.info("Starting live mode. Press 'q' to quit.")
    cap = cv2.VideoCapture(settings.camera_index)
    if not cap.isOpened():
        logger.error(f"Could not open camera {settings.camera_index}")
        return

    stable_frame_count = 0
    last_result = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        doc = detect_document(frame)

        if doc is not None:
            stable_frame_count += 1
            if stable_frame_count >= settings.frame_stability_count:
                result = _process_document(frame)
                last_result = result
                logger.info(f"Live detection: {result.get('doc_type', 'UNKNOWN')} | "
                            f"ID: {result.get('id_number', 'N/A')} | "
                            f"DB: {result.get('db_status', 'N/A')} | "
                            f"Fraud: {result.get('fraud_check', {}).get('fraud_status', 'N/A')}")
                stable_frame_count = 0
        else:
            stable_frame_count = 0

        if last_result and doc is not None:
            overlay_text = f"{last_result.get('doc_type', '')} | {last_result.get('id_number', '')} | {last_result.get('fraud_check', {}).get('fraud_status', '')}"
            cv2.putText(frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("ID Verification - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Live mode stopped.")


def main():
    parser = argparse.ArgumentParser(description="Nigerian ID Verification System")
    parser.add_argument("--mode", required=True, choices=["image", "live"],
                        help="Run mode: 'image' or 'live'")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image (required for image mode)")
    args = parser.parse_args()

    ensure_folder_exists(settings.face_output_dir)
    ensure_folder_exists("logs")

    if args.mode == "image":
        if not args.image:
            parser.error("--image is required for image mode")
        run_image_mode(args.image)
    elif args.mode == "live":
        run_live_mode()


if __name__ == "__main__":
    main()