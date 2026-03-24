"""
util/helpers.py
Unified helper functions for timestamp parsing and field normalization.
"""
import pandas as pd
from datetime import datetime, timezone
from typing import Union, List, Dict
import re

# Field name mapping: original -> normalized
FIELD_MAPPING = {
    'maker/spot_executed_ts': 'maker_spot_executed_ts',
    'taker/swap/haircut_executed_ts': 'taker_swap_haircut_executed_ts',
    'maker/spot_client_order_id': 'maker_spot_client_order_id',
    'taker/swap_client_order_id': 'taker_swap_client_order_id',
    'maker/spot_anticipated_price': 'maker_spot_anticipated_price',
    'maker/spot_executed_qty': 'maker_spot_executed_qty',
    'maker/spot_executed_price': 'maker_spot_executed_price',
    'taker/swap_executed_price': 'taker_swap_executed_price',
    'taker/swap/haircut_executed_qty': 'taker_swap_haircut_executed_qty',
    'anticipated_basis.1': 'anticipated_basis_secondary',
    # Add more mappings as needed
}


def parse_timestamp(ts: Union[str, int, float], unit: str = 'ms') -> pd.Timestamp:
    """
    Parse various timestamp formats to pandas Timestamp (UTC).
    
    Args:
        ts: Timestamp in ISO8601 string, Unix ms, or Unix s
        unit: 'ms' for milliseconds, 's' for seconds
    
    Returns:
        pd.Timestamp in UTC timezone
    """
    if pd.isna(ts):
        return pd.NaT
    
    if isinstance(ts, str):
        # Handle ISO8601 format: '2025-12-27 13:00:00.222000+00:00'
        ts = ts.strip()
        if '+' in ts or ts.endswith('Z'):
            ts = ts.replace('Z', '+00:00')
        return pd.to_datetime(ts, utc=True)
    elif isinstance(ts, (int, float)):
        # Unix timestamp
        if unit == 'ms':
            return pd.to_datetime(ts, unit='ms', utc=True)
        else:
            return pd.to_datetime(ts, unit='s', utc=True)
    return pd.NaT


def normalize_columns(df: pd.DataFrame, mapping: Dict[str, str] = None) -> pd.DataFrame:
    """
    Normalize column names using FIELD_MAPPING.
    
    Args:
        df: Input DataFrame
        mapping: Optional custom mapping dict
    
    Returns:
        DataFrame with normalized column names
    """
    if mapping is None:
        mapping = FIELD_MAPPING
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def extract_symbol_from_path(path: str) -> str:
    """
    Extract symbol from file path like '.../book_ADAUSDT_20251227.csv.gz'
    
    Returns:
        Symbol without USDT suffix, e.g., 'ADA'
    """
    match = re.search(r'(?:book|trade)_([A-Z0-9]+)USDT_', path)
    if match:
        return match.group(1)
    # Fallback: try to extract from filename
    filename = path.split('/')[-1].split('\\')[-1]
    match = re.search(r'([A-Z0-9]+)USDT', filename)
    return match.group(1) if match else None


def ensure_dir(path: str):
    """Ensure directory exists."""
    import os
    os.makedirs(path, exist_ok=True)


def format_float(val: float, precision: int = 10) -> float:
    """Format float to avoid excessive precision."""
    if pd.isna(val):
        return val
    return round(float(val), precision)