"""
util/config.py
Configuration management using pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Project paths
    project_root: Path = Path(__file__).parent.parent
    data_root: Path = project_root / "data"
    dataset_root: Path = project_root / "dataset"
    # market_processed_root: Path = Path(
    #     r"C:\Users\yecha\workspace\kronos_test\Kronos\dataset\market_processed"
    # )
    # bn_trade_root: Path = Path(
    #     r"C:\Users\yecha\workspace\kronos_test\Kronos\dataset\bn_trade"
    # )
    market_processed_root: Path = Path(
        r"dataset\market_processed"
    )
    bn_trade_root: Path = Path(
        r"dataset\bn_trade"
    )    
    # Output paths
    output_root: Path = dataset_root / "preprocessed"
    output_mode0: Path = output_root / "mode0"
    output_mode2: Path = output_root / "mode2"
    
    # Binance API endpoints
    binance_spot_url: str = "https://api.binance.com/api/v3/exchangeInfo"
    binance_swap_url: str = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    
    # Processing parameters
    window_seconds: int = 60  # Backtrack window for feature extraction
    default_ticksize: float = 1e-8  # Fallback ticksize if API fails
    time_tolerance_ms: int = 100  # Allowed timestamp alignment tolerance
    
    # Cache settings
    ticksize_cache_file: Path = project_root / "config" / "ticksize_cache.json"
    ticksize_cache_ttl_hours: int = 24
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = project_root / "logs" / "preprocess.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_config():
    """Get global settings instance."""
    return settings