"""Logging configuration and utilities for Vantara Customer Intelligence Platform."""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Creates or returns a structured logger with standardized formatting.

    Args:
        name: Name of the logger, typically __name__ of the calling module.
        level: Optional log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to value of LOG_LEVEL environment variable or INFO.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level_str = level or os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Do not propagate to root logger to prevent duplicate outputs
        logger.propagate = False

    return logger
