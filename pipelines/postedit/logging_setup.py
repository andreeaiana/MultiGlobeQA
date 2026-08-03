"""Logging configuration for the post-editing pipeline.

Use ``setup_logging(language, log_dir, console_level)`` once at the start of
the main script. After that, any module can simply ``from loguru import
logger`` and use it directly.
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(
    language: str,
    log_dir: Path | None = None,
    console_level: str = "INFO",
) -> Path:
    """Configure loguru with a console sink and a file sink.

    Args:
        language: Language code, used in the log filename.
        log_dir: Directory to write the log file (created if missing). If None,
            defaults to ``output/postedit/logs/``.
        console_level: Log level for stdout (DEBUG, INFO, WARNING, ERROR).

    Returns:
        The path to the file sink that was created.
    """
    if log_dir is None:
        log_dir = Path("output/postedit/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"postedit_{language}_{timestamp}.log"

    # Remove default handler and reconfigure
    logger.remove()
    # Console: clean format, configurable level
    logger.add(
        sys.stdout,
        level=console_level,
        format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}",
        colorize=True,
    )
    # File: full debug, structured, no colors
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention=10,
        encoding="utf-8",
    )

    logger.info(f"Logging to {log_file}")
    return log_file
