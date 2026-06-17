"""
database.py — PostgreSQL interface for the Nigerian ID Verification System.
Handles connection management, identity lookup, and new record insertion.
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
            # Convert row to dict using column names
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
                    confidence, retried
                ) VALUES (
                    %(doc_type)s, %(id_number)s, %(surname)s, %(first_name)s,
                    %(middle_name)s, %(date_of_birth)s, %(issue_date)s,
                    %(expiry_date)s, %(sex)s, %(nationality)s, %(height)s,
                    %(blood_group)s, %(address)s, %(state)s, %(face_path)s,
                    %(confidence)s, %(retried)s
                ) RETURNING *;
                """,
                data
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


# ----------------------------------------------------------------------
# CONNECTION SELF‑TEST – run `python database.py` to verify connectivity
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