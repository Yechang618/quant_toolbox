"""
util/binance_meta.py
Fetch and cache Binance ticksize metadata for spot and swap markets.
"""
import requests
import json
import time
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional
from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)


class TickSizeCache:
    """Simple file-based cache for ticksize data."""
    
    def __init__(self, cache_file: Path, ttl_hours: int = 24):
        self.cache_file = cache_file
        self.ttl_seconds = ttl_hours * 3600
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> Dict:
        """Load cache from file if valid."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            # Check expiry
            if time.time() - data.get('timestamp', 0) < self.ttl_seconds:
                return data.get('data', {})
        except Exception as e:
            logger.warning(f"Failed to load ticksize cache: {e}")
        return {}
    
    def _save_cache(self, data: Dict):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save ticksize cache: {e}")
    
    def get(self, key: str) -> Optional[float]:
        """Get cached value."""
        cache = self._load_cache()
        return cache.get(key)
    
    def set(self, key: str, value: float):
        """Set cached value."""
        cache = self._load_cache()
        cache[key] = value
        self._save_cache(cache)


# Global cache instance
_ticksize_cache = TickSizeCache(
    settings.ticksize_cache_file, 
    settings.ticksize_cache_ttl_hours
)

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
def get_ticksize_pair(symbol: str) -> tuple:
    """
    Get both spot and swap ticksize for a symbol.
    
    Args:
        symbol: Trading pair symbol (with or without USDT)
    
    Returns:
        Tuple of (spot_ticksize, swap_ticksize)
    """
    # Normalize symbol first
    symbol_clean = symbol.upper().strip()
    if symbol_clean.endswith('USDT'):
        symbol_for_api = symbol_clean[:-4]
    else:
        symbol_for_api = symbol_clean
    
    return (
        get_ticksize(symbol_for_api, 'spot'),
        get_ticksize(symbol_for_api, 'swap')
    )