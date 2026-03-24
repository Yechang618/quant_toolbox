"""Binance metadata helpers (tick size fetching & caching).

Provides a simple cached accessor for symbol tick sizes for spot/swap.
"""
from functools import lru_cache
import time
import requests
from pathlib import Path
import json
from typing import Dict

from util.config import settings
from util.helpers import ensure_dir, read_json, write_json


_CACHE_TTL = 24 * 3600  # seconds


def _load_local_cache(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_local_cache(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    write_json(data, path)

# util/binance_meta.py

@lru_cache(maxsize=256)
def get_ticksize(symbol: str, market_type: str) -> float:
    """
    Fetch minimum price increment (ticksize) for a trading pair.
    
    Args:
        symbol: Trading pair symbol (with or without USDT suffix), e.g., 'BTC' or 'BTCUSDT'
        market_type: 'spot' or 'swap'
    
    Returns:
        Ticksize as float, or default fallback value
    """
    # === FIX: Normalize symbol to avoid double USDT ===
    symbol_clean = symbol.upper().strip()
    if symbol_clean.endswith('USDT'):
        symbol_base = symbol_clean[:-4]  # Remove USDT for API query
    else:
        symbol_base = symbol_clean
    full_symbol = f"{symbol_base}USDT"  # Ensure exactly one USDT suffix
    # ==================================================
    
    # Check cache first
    cache_key = f"{full_symbol}_{market_type}"
    cached = _ticksize_cache.get(cache_key)
    if cached is not None:
        return cached
    
    # Fetch from API
    url = settings.binance_spot_url if market_type == 'spot' else settings.binance_swap_url
    
    try:
        response = requests.get(url, params={'symbol': full_symbol}, timeout=10)
        
        # Handle 400 Bad Request (invalid symbol)
        if response.status_code == 400:
            logger.warning(f"Invalid symbol {full_symbol} for {market_type} market, using default ticksize")
            _ticksize_cache.set(cache_key, settings.default_ticksize)
            return settings.default_ticksize
        
        response.raise_for_status()
        data = response.json()
        
        for s in data.get('symbols', []):
            if s['symbol'] == full_symbol:
                for f in s.get('filters', []):
                    if f.get('filterType') == 'PRICE_FILTER':
                        ticksize = float(f['tickSize'])
                        _ticksize_cache.set(cache_key, ticksize)
                        logger.debug(f"Fetched {market_type} ticksize for {full_symbol}: {ticksize}")
                        return ticksize
        
        logger.warning(f"Ticksize not found for {full_symbol} in {market_type} market")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            logger.warning(f"Rate limit hit for Binance API, using cached/default ticksize for {full_symbol}")
        else:
            logger.error(f"HTTP error fetching ticksize for {full_symbol}/{market_type}: {e}")
    except requests.RequestException as e:
        logger.error(f"Network error fetching ticksize for {full_symbol}/{market_type}: {e}")
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Failed to parse ticksize response for {full_symbol}: {e}")
    
    # Fallback to default
    logger.warning(f"Using default ticksize {settings.default_ticksize} for {full_symbol}/{market_type}")
    _ticksize_cache.set(cache_key, settings.default_ticksize)
    return settings.default_ticksize
