"""
Application configuration using Pydantic v2.

All sensitive values (API keys, secrets) are loaded exclusively from
environment variables or a ``.env`` file.

WARNING: Never hard-code API keys or secrets in this file or anywhere else
         in the codebase.  Always use environment variables.

Usage::

    from util.config import settings
    print(settings.BINANCE_API_KEY)
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from environment variables / .env file.

    All fields with ``Optional`` type default to *None* so the application
    can start without credentials (useful for offline/testing scenarios).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Binance API credentials
    # WARNING: Load from environment variables only – never hard-code.
    # ------------------------------------------------------------------
    BINANCE_API_KEY: Optional[str] = Field(
        default=None,
        description="Binance API key (set via BINANCE_API_KEY env var).",
    )
    BINANCE_API_SECRET: Optional[str] = Field(
        default=None,
        description="Binance API secret (set via BINANCE_API_SECRET env var).",
    )

    # ------------------------------------------------------------------
    # Data paths
    # ------------------------------------------------------------------
    DATA_RAW_DIR: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw downloaded data.",
    )
    DATA_PROCESSED_DIR: Path = Field(
        default=Path("data/processed"),
        description="Directory for preprocessed / feature-engineered data.",
    )
    MODEL_DIR: Path = Field(
        default=Path("model"),
        description="Directory for saved model weights.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Optional path to a log file.",
    )

    # ------------------------------------------------------------------
    # Training hyper-parameters (sensible defaults)
    # ------------------------------------------------------------------
    RANDOM_SEED: int = Field(default=42, description="Global random seed.")
    BATCH_SIZE: int = Field(default=64, description="Mini-batch size for training.")
    LEARNING_RATE: float = Field(default=1e-3, description="Initial learning rate.")
    NUM_EPOCHS: int = Field(default=50, description="Maximum training epochs.")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"LOG_LEVEL must be one of {valid}, got {v!r}")
        return upper


settings: Settings = Settings()
"""Module-level singleton – import this in other modules.

Note: If required environment variables are missing, ``Settings`` will still
instantiate (all sensitive fields are ``Optional``), but operations that
depend on those fields (e.g. API calls) will raise errors at runtime rather
than at import time.  Set the variables in a ``.env`` file or your shell
before running any data-download scripts.
"""
