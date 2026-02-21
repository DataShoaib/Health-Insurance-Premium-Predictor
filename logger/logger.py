import logging
from logging.handlers import RotatingFileHandler
import os


def get_logger(
    name: str,
    log_file: str = "info.log",
    level: int = logging.INFO,
):
    """
    Creates and returns a logger with both console and file handlers.
    """

    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate logs
    if logger.handlers:
        return logger

    # ---------- Formatter ----------
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- Console Handler ----------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ---------- File Handler (Rotating) ----------
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ---------- Add handlers ----------
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
