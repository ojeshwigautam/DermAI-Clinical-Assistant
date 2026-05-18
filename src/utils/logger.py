"""
logger.py — Centralized logging for the P11 project.
"""

import logging
import os
import sys
from datetime import datetime


def get_logger(name: str, log_dir: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a logger with console + optional file handlers.

    Args:
        name:    Logger name (typically __name__ of the calling module).
        log_dir: If provided, also writes logs to a timestamped file here.
        level:   Logging level (default INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
