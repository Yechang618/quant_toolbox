"""
data/feature_engineering.py
Core feature engineering logic for slippage research.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from util.helpers import parse_timestamp, format_float
from util.binance_meta import get_ticksize_pair
from util.logger import get_logger

logger = get_logger(__name__)


def calculate_midprice(bid1: float, ask1: float) -> float:
    """Calculate mid price from best bid/ask."""
    if pd.isna(bid1) or pd.isna(ask1):
        return np.nan
    return (bid1 + ask1) / 2


def extract_window_features(
    ob_window: pd.DataFrame,
    tf_window: pd.DataFrame,
    spot_tick: float,
    swap_tick: float,
    trade_record: Dict
) -> Dict[str, float]:
    """
    Extract features from 60-second market data window.
    
    Args:
        ob_window: Orderbook DataFrame filtered to time window
        tf_window: Trade Flow DataFrame filtered to time window
        spot_tick: Spot market ticksize
        swap_tick: Swap market ticksize
        trade_record: Single trade record dict with execution info
    
    Returns:
        Dictionary of feature name -> value
    """
    features = {}
    
    if ob_window.empty:
        logger.warning("Empty orderbook window, returning NaN features")
        return {k: np.nan for k in [
            'midprice_mean', 'midprice_std', 'spread_ticks', 
            'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_60s'
        ]}
    
    # === Price & Spread Features ===
    spot_bid1 = ob_window['spot_bid1_px']
    spot_ask1 = ob_window['spot_ask1_px']
    midprice = (spot_bid1 + spot_ask1) / 2
    spread = spot_ask1 - spot_bid1
    
    features['midprice_mean'] = format_float(midprice.mean())
    features['midprice_std'] = format_float(midprice.std())
    features['spread_mean'] = format_float(spread.mean())
    
    # Ticksize-normalized spread
    avg_tick = (spot_tick + swap_tick) / 2
    features['spread_ticks'] = format_float((spread / spot_tick).mean()) if spot_tick > 0 else np.nan
    
    # === Liquidity Features ===
    spot_bid1_qty = ob_window['spot_bid1_qty']
    spot_ask1_qty = ob_window['spot_ask1_qty']
    
    # Depth imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    depth_imbalance = (spot_bid1_qty - spot_ask1_qty) / (spot_bid1_qty + spot_ask1_qty + 1e-10)
    features['depth_imbalance_mean'] = format_float(depth_imbalance.mean())
    
    # Top-of-book depth in tick units
    features['depth1_bid_ticks'] = format_float(
        (spot_bid1_qty * spot_bid1 / spot_tick).mean() if spot_tick > 0 else np.nan
    )
    features['depth1_ask_ticks'] = format_float(
        (spot_ask1_qty * spot_ask1 / spot_tick).mean() if spot_tick > 0 else np.nan
    )
    
    # === Volatility Features ===
    features['volatility_ticks'] = format_float(
        midprice.std() / avg_tick if avg_tick > 0 and not pd.isna(midprice.std()) else np.nan
    )
    
    # Price return over window
    if len(midprice) >= 2:
        price_return = (midprice.iloc[-1] - midprice.iloc[0]) / (midprice.iloc[0] + 1e-10)
        features['price_return_60s'] = format_float(price_return)
    else:
        features['price_return_60s'] = np.nan
    
    # === Trade Flow Features ===
    if not tf_window.empty:
        features['trade_volume_60s'] = format_float(tf_window['bid_qty'].fillna(0).sum())
        features['trade_count_60s'] = len(tf_window)
        
        # Trade direction imbalance
        if 'trade_type' in tf_window.columns:
            buy_trades = (tf_window['trade_type'] == 'BUY').sum()
            features['buy_trade_ratio'] = format_float(buy_trades / len(tf_window))
    else:
        features['trade_volume_60s'] = 0
        features['trade_count_60s'] = 0
        features['buy_trade_ratio'] = np.nan
    
    # === Execution Context Features ===
    # These come from trade_record, added for model context
    features['execute_delay_ms'] = trade_record.get('execute_delay_ms', np.nan)
    features['threshold'] = trade_record.get('threshold', np.nan)
    features['basis_expected'] = trade_record.get('basis_expected', np.nan)
    features['basis_executed'] = trade_record.get('basis_executed', np.nan)
    
    # Slippage in ticks
    basis_slippage = trade_record.get('basis_slippage', np.nan)
    if not pd.isna(basis_slippage) and avg_tick > 0:
        features['basis_slippage_ticks'] = format_float(basis_slippage / avg_tick)
    else:
        features['basis_slippage_ticks'] = np.nan
    
    return features


def prepare_trade_record_features(record: pd.Series) -> Dict:
    """
    Prepare derived fields from a single trade record.
    
    Args:
        record: Single row from Trade Records DataFrame
    
    Returns:
        Dict with original + derived fields
    """
    result = record.to_dict()
    
    # Calculate execute_delay
    timer_start = record.get('timer_start_ts')
    taker_exec = record.get('taker_swap_haircut_executed_ts')
    
    if pd.notna(timer_start) and pd.notna(taker_exec):
        result['execute_delay_ms'] = int(taker_exec - timer_start)
    else:
        result['execute_delay_ms'] = np.nan
    
    # Basis slippage
    anticipated = record.get('anticipated_basis')
    executed = record.get('executed_basis')
    
    if pd.notna(anticipated) and pd.notna(executed):
        result['basis_slippage'] = executed - anticipated
    else:
        result['basis_slippage'] = np.nan
    
    # Ensure key fields exist
    result['threshold'] = record.get('threshold', np.nan)
    result['basis_expected'] = anticipated
    result['basis_executed'] = executed
    
    return result