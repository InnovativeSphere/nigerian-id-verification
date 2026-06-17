"""
logger.py — Centralized logging configuration for the ID Verification System.
Every module imports get_logger() from here — no one configures their own logger.
"""

import logging
import os
from config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger with the given name.
    Logs are written to both the terminal and the log file.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A standard Python logger ready to use.
    """
    logger = logging.getLogger(name)

    # Only configure handlers once — prevents duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Log format: timestamp | level | message
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Ensure the logs folder exists
        log_dir = os.path.dirname(settings.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # File handler — writes to logs/verification.log
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler — prints to terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger