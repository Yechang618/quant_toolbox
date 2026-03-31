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
        # logger.warning("Empty orderbook window, returning NaN features")
        return {k: np.nan for k in [
            'spot_midprice_mean', 'spot_midprice_std', 'spread_ticks', 
            'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_60s',
            'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low',
            'swap_midprice_mean', 'swap_midprice_std', 'swap_spread_mean', 'swap_spread_ticks',
            'swap_depth_imbalance_mean', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks',
            'swap_volatility_ticks', 'swap_price_return_60s', 'swap_trade_volume_60s', 'swap_trade_count_60s', 'swap_buy_trade_ratio',
            'basis_ask_mean', 'basis_bid_mean', 'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low',
            'spot_buy_trade_ratio', 'execute_delay_ms', 'threshold', 'basis_expected', 'basis_executed', 'spot_basis_slippage_ticks'
        ]}
    
    # === Price & Spread Features ===
    spot_bid1 = ob_window['spot_bid1_px']
    spot_ask1 = ob_window['spot_ask1_px']
    swap_bid1 = ob_window['swap_bid1_px']
    swap_ask1 = ob_window['swap_ask1_px']
    spot_midprice = (spot_bid1 + spot_ask1) / 2
    swap_midprice = (swap_bid1 + swap_ask1) / 2
    spot_spread = spot_ask1 - spot_bid1
    swap_spread = swap_ask1 - swap_bid1
    basis_ask = np.log(swap_ask1) - np.log(spot_ask1)
    basis_bid = np.log(swap_bid1) - np.log(spot_bid1)

    features['spot_midprice_mean'] = format_float(spot_midprice.mean())
    features['spot_midprice_std'] = format_float(spot_midprice.std())
    features['spot_spread_mean'] = format_float(spot_spread.mean())
    features['spot_midprice_open'] = format_float(spot_midprice.iloc[0])
    features['spot_midprice_close'] = format_float(spot_midprice.iloc[-1])
    features['spot_midprice_high'] = format_float(spot_midprice.max())
    features['spot_midprice_low'] = format_float(spot_midprice.min())
    features['swap_midprice_mean'] = format_float(swap_midprice.mean())
    features['swap_midprice_std'] = format_float(swap_midprice.std())
    features['swap_spread_mean'] = format_float(swap_spread.mean())
    features['swap_spread_ticks'] = format_float((swap_spread / swap_tick).mean()) if swap_tick > 0 else np.nan
    # Ticksize-normalized spread
    avg_tick = (spot_tick + swap_tick) / 2
    features['spot_spread_ticks'] = format_float((spot_spread / spot_tick).mean()) if spot_tick > 0 else np.nan

    # Basis features
    features['basis_ask_mean'] = format_float(basis_ask.mean())
    features['basis_bid_mean'] = format_float(basis_bid.mean())
    features['basis_ask_open'] = format_float(basis_ask.iloc[0])
    features['basis_bid_open'] = format_float(basis_bid.iloc[0])
    features['basis_ask_close'] = format_float(basis_ask.iloc[-1])
    features['basis_bid_close'] = format_float(basis_bid.iloc[-1])
    features['basis_ask_high'] = format_float(basis_ask.max())
    features['basis_bid_high'] = format_float(basis_bid.max())
    features['basis_ask_low'] = format_float(basis_ask.min())
    features['basis_bid_low'] = format_float(basis_bid.min())
    
    # === Liquidity Features ===
    spot_bid1_qty = ob_window['spot_bid1_qty']
    spot_ask1_qty = ob_window['spot_ask1_qty']
    swap_bid1_qty = ob_window['swap_bid1_qty']
    swap_ask1_qty = ob_window['swap_ask1_qty']
    
    # Depth imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
    spot_depth_imbalance = (spot_bid1_qty - spot_ask1_qty) / (spot_bid1_qty + spot_ask1_qty + 1e-10)
    features['spot_depth_imbalance_mean'] = format_float(spot_depth_imbalance.mean())
    swap_depth_imbalance = (swap_bid1_qty - swap_ask1_qty) / (swap_bid1_qty + swap_ask1_qty + 1e-10)
    features['swap_depth_imbalance_mean'] = format_float(swap_depth_imbalance.mean())
    
    # Top-of-book depth in tick units
    features['spot_depth1_bid_ticks'] = format_float(
        (spot_bid1_qty * spot_bid1 / spot_tick).mean() if spot_tick > 0 else np.nan
    )
    features['spot_depth1_ask_ticks'] = format_float(
        (spot_ask1_qty * spot_ask1 / spot_tick).mean() if spot_tick > 0 else np.nan
    )
    features['swap_depth1_bid_ticks'] = format_float(
        (swap_bid1_qty * swap_bid1 / swap_tick).mean() if swap_tick > 0 else np.nan
    )
    features['swap_depth1_ask_ticks'] = format_float(
        (swap_ask1_qty * swap_ask1 / swap_tick).mean() if swap_tick > 0 else np.nan
    )
    

    # === Volatility Features ===
    features['spot_volatility_ticks'] = format_float(
        spot_midprice.std() / avg_tick if avg_tick > 0 and not pd.isna(spot_midprice.std()) else np.nan
    )
    features['swap_volatility_ticks'] = format_float(
        swap_midprice.std() / avg_tick if avg_tick > 0 and not pd.isna(swap_midprice.std()) else np.nan 
    )

    # Price return over window
    if len(spot_midprice) >= 2:
        price_return = (spot_midprice.iloc[-1] - spot_midprice.iloc[0]) / (spot_midprice.iloc[0] + 1e-10)
        features['spot_price_return_60s'] = format_float(price_return)
    else:
        features['spot_price_return_60s'] = np.nan
    
    if len(swap_midprice) >= 2:
        swap_price_return = (swap_midprice.iloc[-1] - swap_midprice.iloc[0]) / (swap_midprice.iloc[0] + 1e-10)
        features['swap_price_return_60s'] = format_float(swap_price_return)
    else:
        features['swap_price_return_60s'] = np.nan

    # === Trade Flow Features ===
    if not tf_window.empty:
        features['spot_trade_volume_60s'] = format_float(tf_window['bid_qty'].fillna(0).sum())
        features['spot_trade_count_60s'] = len(tf_window)
        
        # Trade direction imbalance
        if 'trade_type' in tf_window.columns:
            buy_trades = (tf_window['trade_type'] == 'BUY').sum()
            features['spot_buy_trade_ratio'] = format_float(buy_trades / len(tf_window))
    else:
        features['spot_trade_volume_60s'] = 0
        features['spot_trade_count_60s'] = 0
        features['spot_buy_trade_ratio'] = np.nan
    
    if not tf_window.empty and 'swap_qty' in tf_window.columns:
        features['swap_trade_volume_60s'] = format_float(tf_window['swap_qty'].fillna(0).sum())
        features['swap_trade_count_60s'] = len(tf_window)
        if 'trade_type' in tf_window.columns:
            buy_trades = (tf_window['trade_type'] == 'BUY').sum()
            features['swap_buy_trade_ratio'] = format_float(buy_trades / len(tf_window))
    else:
        features['swap_trade_volume_60s'] = 0
        features['swap_trade_count_60s'] = 0
        features['swap_buy_trade_ratio'] = np.nan


    # === Execution Context Features ===
    # These come from trade_record, added for model context
    features['execute_delay_ms'] = trade_record.get('execute_delay_ms', np.nan)
    features['threshold'] = trade_record.get('threshold', np.nan)
    features['basis_expected'] = trade_record.get('basis_expected', np.nan)
    features['basis_executed'] = trade_record.get('basis_executed', np.nan)
    
    # Slippage in ticks
    basis_slippage = trade_record.get('basis_slippage', np.nan)
    if not pd.isna(basis_slippage) and avg_tick > 0:
        features['spot_basis_slippage_ticks'] = format_float(basis_slippage / avg_tick)
    else:
        features['spot_basis_slippage_ticks'] = np.nan
    
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
    
    # gain_vs_threshold = executed - threshold (open)
    # Factor = basis_mean - threshold (open)
    # gain_vs_thres - Factor = executed - basis_mean

    if pd.notna(anticipated) and pd.notna(executed):
        result['basis_slippage'] = executed - anticipated
    else:
        result['basis_slippage'] = np.nan
    
    # Ensure key fields exist
    result['threshold'] = record.get('threshold', np.nan)
    result['basis_expected'] = anticipated
    result['basis_executed'] = executed
    
    return result