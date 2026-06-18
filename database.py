"""
database.py — PostgreSQL interface for the Nigerian ID Verification System.
Handles connection management, identity lookup, new record insertion,
and fraud‑tracking updates.
"""

import psycopg2
from config import settings
from logger import get_logger

logger = get_logger(__name__)


def get_connection() -> psycopg2.extensions.connection:
    """Open and return a new PostgreSQL connection."""
    return psycopg2.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password
    )


def lookup_identity(id_number: str) -> dict | None:
    """
    Search the identities table by ID number.
    Returns the full record as a dict, or None if not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM identities WHERE id_number = %s;",
                (id_number,)
            )
            row = cur.fetchone()
            if row is None:
                return None
            columns = [desc[0] for desc in cur.description]
            return dict(zip(columns, row))
    finally:
        conn.close()


def insert_identity(data: dict) -> dict:
    """
    Insert a new identity record into the database.
    Returns the inserted record as a dict.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO identities (
                    doc_type, id_number, surname, first_name, middle_name,
                    date_of_birth, issue_date, expiry_date, sex, nationality,
                    height, blood_group, address, state, face_path,
                    confidence, retried,
                    fraud_status, trust_score, flag_count, last_flag_reason, last_flagged_at
                ) VALUES (
                    %(doc_type)s, %(id_number)s, %(surname)s, %(first_name)s,
                    %(middle_name)s, %(date_of_birth)s, %(issue_date)s,
                    %(expiry_date)s, %(sex)s, %(nationality)s, %(height)s,
                    %(blood_group)s, %(address)s, %(state)s, %(face_path)s,
                    %(confidence)s, %(retried)s,
                    %(fraud_status)s, %(trust_score)s, %(flag_count)s,
                    %(last_flag_reason)s, %(last_flagged_at)s
                ) RETURNING *;
                """,
                {
                    **data,
                    "fraud_status": data.get("fraud_status", "CLEAN"),
                    "trust_score": data.get("trust_score"),
                    "flag_count": data.get("flag_count", 0),
                    "last_flag_reason": data.get("last_flag_reason"),
                    "last_flagged_at": data.get("last_flagged_at")
                }
            )
            row = cur.fetchone()
            columns = [desc[0] for desc in cur.description]
            conn.commit()
            logger.info(f"New identity inserted: {data.get('id_number')}")
            return dict(zip(columns, row))
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Database insert failed: {e}")
        raise
    finally:
        conn.close()


def update_fraud_record(id_number: str, fraud_result: dict) -> dict | None:
    """
    Update the fraud‑tracking columns for an existing identity record.
    Increments flag_count if the status is FLAGGED or SUSPICIOUS.
    Returns the updated record or None if the ID number is not found.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, flag_count FROM identities WHERE id_number = %s;",
                (id_number,)
            )
            row = cur.fetchone()
            if row is None:
                return None

            identity_id, current_flag_count = row
            new_flag_count = current_flag_count
            if fraud_result.get("fraud_status") in ("FLAGGED", "SUSPICIOUS"):
                new_flag_count += 1

            cur.execute(
                """
                UPDATE identities
                SET fraud_status = %s,
                    trust_score = %s,
                    flag_count = %s,
                    last_flag_reason = %s,
                    last_flagged_at = NOW()
                WHERE id = %s
                RETURNING *;
                """,
                (
                    fraud_result.get("fraud_status", "CLEAN"),
                    fraud_result.get("trust_score"),
                    new_flag_count,
                    ", ".join(fraud_result.get("issues_detected", [])) or None,
                    identity_id
                )
            )
            updated_row = cur.fetchone()
            columns = [desc[0] for desc in cur.description]
            conn.commit()
            logger.info(f"Fraud record updated for {id_number}: {fraud_result.get('fraud_status')}")
            return dict(zip(columns, updated_row))
    except psycopg2.Error as e:
        conn.rollback()
        logger.error(f"Failed to update fraud record: {e}")
        return None
    finally:
        conn.close()


# ----------------------------------------------------------------------
# CONNECTION SELF‑TEST
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM identities;")
            count = cur.fetchone()[0]
            print(f"✅ Database connection successful. {count} record(s) found.")
        conn.close()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")