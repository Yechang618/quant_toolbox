"""
util/logger.py
Unified logging configuration.
"""
import logging
import sys
from pathlib import Path
from util.config import settings


def setup_logger(name: str, level: str = None, log_file: Path = None) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level string, e.g., 'INFO'
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    level = level or settings.log_level
    log_file = log_file or settings.log_file
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger with default settings."""
    return setup_logger(name)