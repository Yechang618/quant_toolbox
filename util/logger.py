"""
Unified logging configuration for quant_toolbox.

Call :func:`get_logger` from any module to obtain a properly configured logger.
The root logger is configured once (on first import) so that all child loggers
inherit the same format and level.
"""

import logging
import sys
from typing import Optional


_ROOT_CONFIGURED: bool = False
_DEFAULT_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def configure_root_logger(
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATE_FORMAT,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger with a consistent format.

    This function is idempotent: if the root logger has already been configured
    by a previous call it returns immediately.

    Args:
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
        fmt: Log message format string.
        datefmt: Date/time format string.
        log_file: Optional path to a file where logs will also be written.
            If *None*, logs are written to *stderr* only.
    """
    global _ROOT_CONFIGURED  # noqa: PLW0603
    if _ROOT_CONFIGURED:
        return

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _ROOT_CONFIGURED = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a named logger, ensuring the root logger is configured.

    Args:
        name: Logger name, typically ``__name__``.
        level: Optional override level for this specific logger.

    Returns:
        A :class:`logging.Logger` instance.

    Example::

        from util.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello, world!")
    """
    configure_root_logger()
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    return log
