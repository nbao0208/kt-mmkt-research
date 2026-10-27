"""
Logging Utility for Multimodal Knowledge Tracing Experiments.
Configures structured console and file handlers with customizable log levels and formatting.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str = "kt_research",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger with both console and optional file handlers.

    Args:
        name: Name identifier for the logger instance.
        log_file: Optional file path to persist log messages.
        level: Logging verbosity level (default: logging.INFO).

    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was previously instantiated
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Standard output handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optional persistent file handler
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
