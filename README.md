
```markdown
# Nigerian ID Verification System (V2)

A fully automated identity verification pipeline for Nigerian national documents,
now with **fraud / tampering detection** that analyses internal consistency,
font signatures, photo‑splice evidence and ID‑number format validation.

Given a static image or a live webcam feed, the system:

- Detects and straightens the ID card using contour analysis
- Classifies the document type (NIN Card, NIN Slip, Driver's License, Voter's Card, Passport)
- Extracts structured fields (name, ID number, date of birth, sex, etc.) via EasyOCR
- Detects and extracts the facial photo using a Haar cascade
- **Runs three independent fraud checks:**
  - **Font consistency** – catches digitally altered text fields
  - **Photo splice detection** – finds pasted‑in face photographs
  - **ID number format / checksum validation** – flags impossible numbers
- Calculates a **trust score** (0‑100) and assigns a verdict: `CLEAN`, `FLAGGED`, or `SUSPICIOUS`
- Stores or retrieves the identity in a **PostgreSQL** database
- **Tracks repeat fraud attempts** per ID number across multiple scans
- Returns a structured JSON result with full reasoning and logs every verification

The system is inspired by real‑world KYC (Know Your Customer) pipelines used by banks,
fintechs, and government agencies.  Every fraud flag comes with a human‑readable
explanation — never a bare `true`/`false`.

## Features

### V1 (base)
- 5 Nigerian document types
- Contour‑based document detection + perspective correction
- Two‑stage classification (aspect ratio for NIN Slip + OCR keywords)
- Keyword‑based field extraction with automatic retry
- Face extraction (Haar cascade) with padding
- PostgreSQL backend with full CRUD
- Live webcam mode with stability counter
- Audit logging

### V2 (fraud detection)
- **Font consistency analysis** – compares stroke width, edge sharpness and character spacing across fields on the same card
- **Photo splice detection** – uses LBP noise texture, edge‑discontinuity scanning and lighting‑direction comparison
- **ID number validation** – checks length, character type and known patterns per document type
- **Trust score** – weighted combination of all checks, 0‑100
- **Tiered verdict** – `CLEAN`, `FLAGGED` (single issue → review), `SUSPICIOUS` (multiple issues → strong signal)
- **Repeat‑attempt tracking** – `flag_count` increments each time the same ID is flagged

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
  "db_status": "NEW_ENTRY",
  "fraud_check": {
    "fraud_status": "CLEAN",
    "trust_score": 100,
    "issues_detected": [],
    "font_analysis": { "consistent": true },
    "splice_analysis": { "splice_suspected": false },
    "format_validation": { "valid_format": true }
  }
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
   - Run the schema below
   - Update your `.env` file with the database credentials

6. **Configure environment variables**
   - Copy `.env.example` to `.env`
   - Fill in your database credentials and any custom paths

## Database Schema (V2 Extended)

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
    -- V2 fraud‑tracking columns
    fraud_status VARCHAR(20) DEFAULT 'CLEAN',
    trust_score INTEGER,
    flag_count INTEGER DEFAULT 0,
    last_flag_reason TEXT,
    last_flagged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_id_number ON identities(id_number);
CREATE INDEX IF NOT EXISTS idx_doc_type ON identities(doc_type);
```

Run this once to upgrade an existing V1 database:
```sql
ALTER TABLE identities ADD COLUMN IF NOT EXISTS fraud_status VARCHAR(20) DEFAULT 'CLEAN';
ALTER TABLE identities ADD COLUMN IF NOT EXISTS trust_score INTEGER;
ALTER TABLE identities ADD COLUMN IF NOT EXISTS flag_count INTEGER DEFAULT 0;
ALTER TABLE identities ADD COLUMN IF NOT EXISTS last_flag_reason TEXT;
ALTER TABLE identities ADD COLUMN IF NOT EXISTS last_flagged_at TIMESTAMP;
```

## Configuration

All runtime settings are loaded from a `.env` file via `python-dotenv` and validated with Pydantic.

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

## Usage

### Image mode – process a single photo
```bash
python main.py --mode image --image path/to/id.jpg
```

### Live mode – real‑time webcam verification
```bash
python main.py --mode live
```
Hold the ID card steady for a few moments.  The system waits for a stable frame
before running the full pipeline.  Press `q` to quit.

## Project Structure

```
nigerian-id-verification/
├── main.py               # Orchestrator (image & live modes)
├── detector.py           # Document detection + perspective transform
├── classifier.py         # Two‑stage document classification
├── extractor.py          # Field extraction with retry + region mapping
├── reader.py             # Centralized EasyOCR wrapper
├── face.py               # Haar cascade face extraction
├── database.py           # PostgreSQL connection, lookup, insert, fraud update
├── logger.py             # Centralized logging setup
├── utils.py              # Preprocessing, folder management, saving
├── config.py             # Pydantic settings (not committed)
├── config.example.py     # Template config for GitHub
│
├── fraud_check.py        # V2: fraud orchestrator
├── font_analysis.py      # V2: font consistency check
├── splice_detection.py   # V2: photo overlay / splice detection
├── id_validation.py      # V2: ID number format & checksum validation
│
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
- scikit-image

All listed in `requirements.txt`.

## Fraud Detection Details

### Font consistency
Compares stroke width, Laplacian edge sharpness and character‑spacing regularity
across every OCR‑located field on the card.  Any field that deviates significantly
from the card's own median profile is flagged as a possible digital edit.

### Photo splice detection
Analyses the face region and its surrounding background:
- **Noise texture** – LBP histograms compared via Chi‑squared distance
- **Edge artifacts** – elevated Canny edge density along the face boundary
- **Lighting direction** – gradient‑based angle comparison between face and background

Requires **two of three** signals to agree before flagging, reducing false positives.

### ID number validation
Uses publicly documented format rules for each Nigerian ID type:
- NIN: exactly 11 digits
- Driver's License: alphanumeric, typical pattern `LL…DD…LL`
- Voter's Card: alphanumeric, 6‑19 characters
- Passport: one letter + 8 digits

## Future Upgrades (V3)

- **Face matching** – compare ID photo against a live webcam image of the presenter
- **YOLO document detection** – replace contour analysis with a deep‑learning model
- **Cross‑document consistency** – verify names / DOB across multiple ID types
- **Web dashboard** – FastAPI + real‑time monitoring UI
- **Machine‑learning splice detector** – trained on a synthetic tampered‑ID dataset

---