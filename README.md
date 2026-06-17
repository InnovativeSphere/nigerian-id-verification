
```markdown
# Nigerian ID Verification System

A fully automated identity verification pipeline for Nigerian national documents.  
Given a static image or a live webcam feed, the system:

- Detects and straightens the ID card using contour analysis
- Classifies the document type (NIN Card, NIN Slip, Driver's License, Voter's Card, Passport)
- Extracts structured fields (name, ID number, date of birth, sex, etc.) via EasyOCR
- Detects and extracts the facial photo using a Haar cascade
- Stores or retrieves the identity in a **PostgreSQL** database
- Returns a structured JSON result and logs every verification for audit

The system is inspired by real‑world KYC (Know Your Customer) pipelines used by banks, fintechs, and government agencies.

## Features

- **5 Nigerian document types** – NIN Card, NIN Slip, Driver's License (FRSC), Voter's Card / PVC (INEC), International Passport
- **Contour‑based document detection** with perspective correction
- **Two‑stage classification** – aspect ratio (for the wide NIN Slip) + OCR keyword matching
- **Keyword‑based field extraction** with automatic retry on poor reads
- **Face extraction** (Haar cascade) with generous padding
- **PostgreSQL backend** – full CRUD, with indexes for fast lookup
- **Live webcam mode** with a stability counter (prevents blurry mid‑motion OCR)
- **Audit logging** – every detection is timestamped and written to a log file
- **Fully typed configuration** via Pydantic + `.env` (no hardcoded secrets)

## Demo

```bash
$ python main.py --mode image --image test_images/fake_nin_card.jpg
{
  "doc_type": "NIN_CARD",
  "id_number": "23456789012",
  "surname": "BELLO",
  "first_name": "AMINA",
  "nationality": "NGA",
  "confidence": "HIGH",
  "db_status": "NEW_ENTRY"
}
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/InnovativeSphere/nigerian-id-verification.git
   cd nigerian-id-verification
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Haar cascade for face detection**
   ```bash
   mkdir cascades
   curl -o cascades/haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
   ```
   (On Windows PowerShell, use `Invoke-WebRequest -Uri "..." -OutFile "cascades/haarcascade_frontalface_default.xml"`)

5. **Set up PostgreSQL**
   - Create a database (e.g., `id_verification`)
   - Run the schema from `schema.sql` (see below)
   - Update your `.env` file with the database credentials

6. **Configure environment variables**
   - Copy `.env.example` to `.env` (or create a new one)
   - Fill in your database credentials and any custom paths

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS identities (
    id SERIAL PRIMARY KEY,
    doc_type VARCHAR(20) NOT NULL,
    id_number VARCHAR(50) UNIQUE NOT NULL,
    surname VARCHAR(100) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    middle_name VARCHAR(100),
    date_of_birth DATE,
    issue_date DATE,
    expiry_date DATE,
    sex CHAR(1),
    nationality VARCHAR(50),
    height VARCHAR(10),
    blood_group VARCHAR(5),
    address TEXT,
    state VARCHAR(50),
    face_path VARCHAR(255),
    confidence VARCHAR(10),
    retried BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_id_number ON identities(id_number);
CREATE INDEX IF NOT EXISTS idx_doc_type ON identities(doc_type);
```

## Configuration

All runtime settings are loaded from a `.env` file via `python-dotenv` and validated with Pydantic.  
Never hardcode secrets – the `.env` file is gitignored.

### Example `.env` file
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=id_verification
DB_USER=postgres
DB_PASSWORD=your_password_here
CASCADE_PATH=cascades/haarcascade_frontalface_default.xml
FACE_OUTPUT_DIR=extracted_faces
LOG_FILE=logs/verification.log
FRAME_STABILITY_COUNT=10
```

You can also adjust `camera_index`, `detection_confidence`, and other parameters in `config.py` (all with Pydantic defaults).

## Usage

### Image mode – process a single photo
```bash
python main.py --mode image --image path/to/id.jpg
```

### Live mode – real‑time webcam verification
```bash
python main.py --mode live
```
Hold the ID card steady for a few moments. The system waits for a stable frame before running the full pipeline. Press `q` to quit.

## Project Structure

```
nigerian-id-verification/
├── main.py               # Orchestrator (image & live modes)
├── detector.py           # Document detection + perspective transform
├── classifier.py         # Two‑stage document classification
├── extractor.py          # Field extraction with retry logic
├── reader.py             # Centralized EasyOCR wrapper
├── face.py               # Haar cascade face extraction
├── database.py           # PostgreSQL connection, lookup, insert
├── logger.py             # Centralized logging setup
├── utils.py              # Preprocessing, folder management, saving
├── config.py             # Pydantic settings (not committed)
├── config.example.py     # Template config for GitHub
├── cascades/             # Haar cascade XML file
├── extracted_faces/      # Saved face images (gitignored)
├── logs/                 # Verification logs (gitignored)
├── test_images/          # Sample / synthetic ID images
├── requirements.txt
└── README.md
```

## Dependencies

- opencv-python
- easyocr
- torch
- psycopg2-binary
- python-dotenv
- pydantic
- Pillow
- numpy

All listed in `requirements.txt`.

## Future Upgrades (V2)

- **Face matching** – compare the face on the ID with a live camera feed of the person presenting it
- **Deep‑learning detection** – swap contour detection for a YOLO model to handle more angles
- **Web dashboard** – a FastAPI front‑end for real‑time monitoring and plate management
- **Multi‑camera support** – monitor multiple inspection stations simultaneously

---
