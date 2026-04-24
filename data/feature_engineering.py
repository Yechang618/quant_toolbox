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


# def extract_window_features(
#     ob_window: pd.DataFrame,
#     tf_window: pd.DataFrame,
#     spot_tick: float,
#     swap_tick: float,
#     trade_record: Dict
# ) -> Dict[str, float]:
#     """
#     Extract features from 60-second market data window.
    
#     Args:
#         ob_window: Orderbook DataFrame filtered to time window
#         tf_window: Trade Flow DataFrame filtered to time window
#         spot_tick: Spot market ticksize
#         swap_tick: Swap market ticksize
#         trade_record: Single trade record dict with execution info
    
#     Returns:
#         Dictionary of feature name -> value
#     """
#     features = {}
    
#     if ob_window.empty:
#         # logger.warning("Empty orderbook window, returning NaN features")
#         feature_list = [
#             'basis_ask_mean', 'basis_bid_mean', 'basis_ask_std', 'basis_bid_std', 
#             'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
#             'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low',
#             'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 'basis_ask_adjusted_std', 'basis_bid_adjusted_std',
#             'spot_spread_ticks', 'spot_spread_mean', 'swap_spread_ticks', 'swap_spread_mean', 
#             'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean',
#             'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks',
#             'spot_volatility_ticks', 'swap_volatility_ticks', 'spot_price_return_2min', 'swap_price_return_2min',
#             'spot_trade_volume_2min', 'spot_trade_count_2min', 'spot_bid_price_mean', 'spot_ask_price_mean',
#             'swap_bid_price_mean', 'swap_ask_price_mean', 'swap_trade_volume_2min', 'swap_trade_count_2min',
#             'spot_buy_trade_ratio', 'execute_delay_ms', 'threshold', 'basis_expected', 'basis_executed', 'basis_slippage_ticks',
#             'spot_midprice_mean', 'spot_midprice_std', 'spread_ticks', 
#             'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_2min',
#             'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low',
#             'swap_midprice_mean', 'swap_midprice_std', 'swap_spread_mean', 
#             'swap_depth_imbalance_mean', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks', 'swap_buy_trade_ratio',
#         ]
#         return {k: np.nan for k in feature_list}
#     # print(f"ob_window columns: {ob_window.columns}")
#     # ob_window columns: Index(['spot_exchange', 'spot_stream', 'symbol_x', 'time_str',
#     #    'spot_event_time', 'spot_trade_time', 'spot_bid1_px', 'spot_bid1_qty',
#     #    'spot_bid2_px', 'spot_bid2_qty', 'spot_bid3_px', 'spot_bid3_qty',
#     #    'spot_bid4_px', 'spot_bid4_qty', 'spot_bid5_px', 'spot_bid5_qty',
#     #    'spot_ask1_px', 'spot_ask1_qty', 'spot_ask2_px', 'spot_ask2_qty',
#     #    'spot_ask3_px', 'spot_ask3_qty', 'spot_ask4_px', 'spot_ask4_qty',
#     #    'spot_ask5_px', 'spot_ask5_qty', 'swap_exchange', 'swap_stream',
#     #    'symbol_y', 'swap_event_time', 'swap_trade_time', 'swap_bid1_px',
#     #    'swap_bid1_qty', 'swap_bid2_px', 'swap_bid2_qty', 'swap_bid3_px',
#     #    'swap_bid3_qty', 'swap_bid4_px', 'swap_bid4_qty', 'swap_bid5_px',
#     #    'swap_bid5_qty', 'swap_ask1_px', 'swap_ask1_qty', 'swap_ask2_px',
#     #    'swap_ask2_qty', 'swap_ask3_px', 'swap_ask3_qty', 'swap_ask4_px',
#     #    'swap_ask4_qty', 'swap_ask5_px', 'swap_ask5_qty', 'funding_rate',
#     #    'index_price', 'ts_ms'],
#     #   dtype='str')

#     # tf_window columns: Index(['exchange', 'stream', 'symbol', 'local_ts', 'time_str', 'update_id',
#     #    'event_time', 'trade_time', 'bid_px', 'bid_qty', 'ask_px', 'ask_qty',
#     #    'bid1_px', 'bid1_qty', 'bid2_px', 'bid2_qty', 'bid3_px', 'bid3_qty',
#     #    'bid4_px', 'bid4_qty', 'bid5_px', 'bid5_qty', 'ask1_px', 'ask1_qty',
#     #    'ask2_px', 'ask2_qty', 'ask3_px', 'ask3_qty', 'ask4_px', 'ask4_qty',
#     #    'ask5_px', 'ask5_qty', 'trade_type', 'ts_ms'],
#     #   dtype='str')

#     # === Price & Spread Features ===

#     # Order book
#     spot_bid1 = ob_window['spot_bid1_px']
#     spot_ask1 = ob_window['spot_ask1_px']
#     swap_bid1 = ob_window['swap_bid1_px']
#     swap_ask1 = ob_window['swap_ask1_px']
#     spot_midprice = (spot_bid1 + spot_ask1) / 2
#     swap_midprice = (swap_bid1 + swap_ask1) / 2
#     spot_spread = spot_ask1 - spot_bid1
#     swap_spread = swap_ask1 - swap_bid1
    
#     basis_ask = np.log(swap_ask1) - np.log(spot_ask1)
#     basis_bid = np.log(swap_bid1) - np.log(spot_bid1)
#     basis_ask_bid = np.log(swap_ask1) - np.log(spot_bid1)
#     basis_bid_ask = np.log(swap_bid1) - np.log(spot_ask1)
#     ob_window['basis_ask'] = basis_ask
#     ob_window['basis_bid'] = basis_bid
#     ob_window['basis_ask_bid'] = basis_ask_bid
#     ob_window['basis_bid_ask'] = basis_bid_ask
#     # a = ob_window[['basis_ask', 'basis_ask_bid']].values()
#     # print(f"basis_ask and basis_ask_bid shape: {a.shape}")
#     # print(a)
#     basis_ask_mix = np.concatenate([basis_ask.values.reshape(-1), basis_ask_bid.values.reshape(-1)])
#     # ob_window[['basis_ask', 'basis_ask_bid']].values().reshape(-1)
#     basis_bid_mix = np.concatenate([basis_bid.values.reshape(-1), basis_bid_ask.values.reshape(-1)])
#     # ob_window[['basis_bid', 'basis_bid_ask']].values().reshape(-1)
#     basis_mid = np.log(swap_midprice) - np.log(spot_midprice)
#     ob_window['basis_ask'] = basis_ask
#     ob_window['basis_bid'] = basis_bid
#     ob_window['basis_ask_bid'] = basis_ask_bid
#     ob_window['basis_bid_ask'] = basis_bid_ask    

#     # if trade_record.get('mode') == 2:
#     #     if trade_record.get('operation') == 'open2':
#     #         upper_bound = np.percentile(basis_bid_mix, 90)
#     #         basis_capped =  basis_bid_mix.where(basis_bid_mix < upper_bound).dropna()
#     #     else:
#     #         lower_bound = np.percentile(basis_ask_mix, 10)
#     #         basis_capped = np.basis_ask_mix.where(basis_ask_mix > lower_bound).dropna()
#     # else:
#     #     if trade_record.get('operation') == 'open2':
#     #         upper_bound = np.percentile(basis_ask_mix, 90)
#     #         basis_capped =  basis_ask_mix.where(basis_ask_mix < upper_bound).dropna()
#     #     else:
#     #         lower_bound = np.percentile(basis_bid_mix, 10)
#     #         basis_capped = basis_bid_mix.where(basis_bid_mix > lower_bound).dropna()

#     # 1. 确定使用哪个序列 & 计算分位数
#     if trade_record.get('mode') == 2:
#         use_bid = trade_record.get('operation') == 'open2'
#     else:
#         use_bid = trade_record.get('operation') != 'open2'

#     series = basis_bid_mix if use_bid else basis_ask_mix
#     percentile = 90 if use_bid else 10
#     bound = np.percentile(series, percentile)

#     # 2. 选择处理方式（二选一）

#     # 🅰️ 过滤模式（删除异常值，长度改变）
#     basis_capped = series[series < bound] if use_bid else series[series > bound]

#     # Weighted basis adjustment based on operation type
#     weights = [-1e6, -1e5, -1e4, -1e3, -1e2, -10, -1, 0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
#     for i, w in enumerate(weights):
#         wght = np.exp(w * basis_mid)/np.sum(np.exp(w * basis_mid))  # Normalize weights to keep scale
#         basis_mid_weighted = basis_mid * wght
#         if w != 0:
#             features[f'basis_mid_weighted_mean_{np.sign(w)}{np.log10(np.abs(w))}'] = format_float(basis_mid_weighted.mean())
#             features[f'basis_mid_weighted_std_{np.sign(w)}{np.log10(np.abs(w))}'] = format_float(basis_mid_weighted.std())
#         else:
#             features[f'basis_mid_weighted_mean_zero'] = format_float(basis_mid_weighted.mean())
#             features[f'basis_mid_weighted_std_zero'] = format_float(basis_mid_weighted.std())            

#     features['spot_midprice_mean'] = format_float(spot_midprice.mean())
#     features['spot_midprice_std'] = format_float(spot_midprice.std())
#     features['spot_spread_mean'] = format_float(spot_spread.mean())
#     features['spot_midprice_open'] = format_float(spot_midprice.iloc[0])
#     features['spot_midprice_close'] = format_float(spot_midprice.iloc[-1])
#     features['spot_midprice_high'] = format_float(spot_midprice.max())
#     features['spot_midprice_low'] = format_float(spot_midprice.min())
#     features['swap_midprice_mean'] = format_float(swap_midprice.mean())
#     features['swap_midprice_std'] = format_float(swap_midprice.std())
#     features['swap_spread_mean'] = format_float(swap_spread.mean())
#     features['swap_spread_ticks'] = format_float((swap_spread / swap_tick).mean()) if swap_tick > 0 else np.nan
#     # Ticksize-normalized spread
#     avg_tick = (spot_tick + swap_tick) / 2
#     features['spot_spread_ticks'] = format_float((spot_spread / spot_tick).mean()) if spot_tick > 0 else np.nan

#     # Basis features
#     features['basis_ask_mean'] = format_float(basis_ask.mean())
#     features['basis_bid_mean'] = format_float(basis_bid.mean())
#     features['basis_ask_std'] = format_float(basis_ask.std())
#     features['basis_bid_std'] = format_float(basis_bid.std())
#     features['basis_ask_open'] = format_float(basis_ask.iloc[0])
#     features['basis_bid_open'] = format_float(basis_bid.iloc[0])
#     features['basis_ask_close'] = format_float(basis_ask.iloc[-1])
#     features['basis_bid_close'] = format_float(basis_bid.iloc[-1])
#     features['basis_ask_high'] = format_float(basis_ask.max())
#     features['basis_bid_high'] = format_float(basis_bid.max())
#     features['basis_ask_low'] = format_float(basis_ask.min())
#     features['basis_bid_low'] = format_float(basis_bid.min())
#     # features['basis_ask_adjusted_mean'] = format_float(basis_ask_adjusted.mean())
#     # features['basis_bid_adjusted_mean'] = format_float(basis_bid_adjusted.mean())
#     # features['basis_ask_adjusted_std'] = format_float(basis_ask_adjusted.std())
#     # features['basis_bid_adjusted_std'] = format_float(basis_bid_adjusted.std())
#     features['basis_mid_adjusted_mean'] = format_float(np.mean(basis_capped))
#     features['basis_mid_adjusted_std'] = format_float(np.std(basis_capped))


#     # === Liquidity Features ===
#     spot_bid1_qty = ob_window['spot_bid1_qty']
#     spot_ask1_qty = ob_window['spot_ask1_qty']
#     swap_bid1_qty = ob_window['swap_bid1_qty']
#     swap_ask1_qty = ob_window['swap_ask1_qty']
#     spot_bid5_qty = ob_window['spot_bid1_qty'] + ob_window['spot_bid2_qty'] + ob_window['spot_bid3_qty'] + ob_window['spot_bid4_qty'] + ob_window['spot_bid5_qty']
#     spot_ask5_qty = ob_window['spot_ask1_qty'] + ob_window['spot_ask2_qty'] + ob_window['spot_ask3_qty'] + ob_window['spot_ask4_qty'] + ob_window['spot_ask5_qty']
#     swap_bid5_qty = ob_window['swap_bid1_qty'] + ob_window['swap_bid2_qty'] + ob_window['swap_bid3_qty'] + ob_window['swap_bid4_qty'] + ob_window['swap_bid5_qty']
#     swap_ask5_qty = ob_window['swap_ask1_qty'] + ob_window['swap_ask2_qty'] + ob_window['swap_ask3_qty'] + ob_window['swap_ask4_qty'] + ob_window['swap_ask5_qty']

#     # Depth imbalance: (bid_qty - ask_qty) / (bid_qty + ask_qty)
#     spot_depth_imbalance = (spot_bid1_qty - spot_ask1_qty) / (spot_bid1_qty + spot_ask1_qty + 1e-10)
#     features['spot_depth_imbalance_mean'] = format_float(spot_depth_imbalance.mean())
#     swap_depth_imbalance = (swap_bid1_qty - swap_ask1_qty) / (swap_bid1_qty + swap_ask1_qty + 1e-10)
#     features['swap_depth_imbalance_mean'] = format_float(swap_depth_imbalance.mean())
#     spot_depth5_imbalance = (spot_bid5_qty - spot_ask5_qty) / (spot_bid5_qty + spot_ask5_qty + 1e-10)
#     features['spot_depth5_imbalance_mean'] = format_float(spot_depth5_imbalance.mean())
#     swap_depth5_imbalance = (swap_bid5_qty - swap_ask5_qty) / (swap_bid5_qty + swap_ask5_qty + 1e-10)
#     features['swap_depth5_imbalance_mean'] = format_float(swap_depth5_imbalance.mean())

#     # Top-of-book depth in tick units
#     features['spot_depth1_bid_ticks'] = format_float(
#         (spot_bid1_qty * spot_bid1 / spot_tick).mean() if spot_tick > 0 else np.nan
#     )
#     features['spot_depth1_ask_ticks'] = format_float(
#         (spot_ask1_qty * spot_ask1 / spot_tick).mean() if spot_tick > 0 else np.nan
#     )
#     features['swap_depth1_bid_ticks'] = format_float(
#         (swap_bid1_qty * swap_bid1 / swap_tick).mean() if swap_tick > 0 else np.nan
#     )
#     features['swap_depth1_ask_ticks'] = format_float(
#         (swap_ask1_qty * swap_ask1 / swap_tick).mean() if swap_tick > 0 else np.nan
#     )
    

#     # === Volatility Features ===
#     features['spot_volatility_ticks'] = format_float(
#         spot_midprice.std() / avg_tick if avg_tick > 0 and not pd.isna(spot_midprice.std()) else np.nan
#     )
#     features['swap_volatility_ticks'] = format_float(
#         swap_midprice.std() / avg_tick if avg_tick > 0 and not pd.isna(swap_midprice.std()) else np.nan 
#     )

#     # Price return over window
#     if len(spot_midprice) >= 2:
#         price_return = (spot_midprice.iloc[-1] - spot_midprice.iloc[0]) / (spot_midprice.iloc[0] + 1e-10)
#         features['spot_price_return_2min'] = format_float(price_return)
#     else:
#         features['spot_price_return_2min'] = np.nan
    
#     if len(swap_midprice) >= 2:
#         swap_price_return = (swap_midprice.iloc[-1] - swap_midprice.iloc[0]) / (swap_midprice.iloc[0] + 1e-10)
#         features['swap_price_return_2min'] = format_float(swap_price_return)
#     else:
#         features['swap_price_return_2min'] = np.nan

#     # # === Trade Flow Features ===
#     if not tf_window.empty:
#         bid_qty_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'bid_qty'].fillna(0)
#         bid_px_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'bid_px'].fillna(0)
#         bid_px_qty_spot = bid_px_spot * bid_qty_spot
#         ask_qty_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'ask_qty'].fillna(0)
#         ask_px_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'ask_px'].fillna(0)
#         ask_px_qty_spot = ask_px_spot * ask_qty_spot   
#         bid_qty_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'bid_qty'].fillna(0)
#         bid_px_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'bid_px'].fillna(0)
#         bid_px_qty_swap = bid_px_swap * bid_qty_swap
#         ask_qty_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'ask_qty'].fillna(0)
#         ask_px_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'ask_px'].fillna(0)
#         ask_px_qty_swap = ask_px_swap * ask_qty_swap
#         features['spot_trade_volume_2min'] = format_float((bid_qty_spot + ask_qty_spot).sum())
#         features['spot_bid_price_mean'] = format_float(bid_px_qty_spot.sum() / (bid_qty_spot.sum() + 1e-10)) if bid_qty_spot.sum() > 0 else np.nan
#         features['spot_ask_price_mean'] = format_float(ask_px_qty_spot.sum() / (ask_qty_spot.sum() + 1e-10)) if ask_qty_spot.sum() > 0 else np.nan
#         features['spot_trade_count_2min'] = len(tf_window.loc[tf_window['trade_type'] == 'spot'])
#         features['spot_buy_trade_ratio'] = format_float(np.log(bid_px_qty_spot.sum() + 1e-10) - np.log(ask_px_qty_spot.sum() + 1e-10)) if ask_px_qty_spot.sum()*bid_px_qty_spot.sum() > 0 else np.nan
#         features['swap_trade_volume_2min'] = format_float((bid_qty_swap + ask_qty_swap).sum())
#         features['swap_trade_count_2min'] = len(tf_window.loc[tf_window['trade_type'] == 'swap'])
#         features['swap_bid_price_mean'] = format_float(bid_px_qty_swap.sum() / (bid_qty_swap.sum() + 1e-10)) if bid_qty_swap.sum() > 0 else np.nan
#         features['swap_ask_price_mean'] = format_float(ask_px_qty_swap.sum() / (ask_qty_swap.sum() + 1e-10)) if ask_qty_swap.sum() > 0 else np.nan
#         features['swap_buy_trade_ratio'] = format_float(np.log(bid_px_qty_swap.sum() + 1e-10) - np.log(ask_px_qty_swap.sum() + 1e-10)) if bid_px_qty_swap.sum()*ask_px_qty_swap.sum() > 0 else np.nan

#     # if not tf_window.empty:
#     #     features['spot_trade_volume_2min'] = format_float(tf_window['bid_qty'].fillna(0).sum())
#     #     features['spot_trade_count_2min'] = len(tf_window)
        
#     #     # Trade direction imbalance
#     #     if 'trade_type' in tf_window.columns:
#     #         buy_trades = (tf_window['trade_type'] == 'BUY').sum()
#     #         features['spot_buy_trade_ratio'] = format_float(buy_trades / len(tf_window))
#     # else:
#     #     features['spot_trade_volume_2min'] = 0
#     #     features['spot_trade_count_2min'] = 0
#     #     features['spot_buy_trade_ratio'] = np.nan
    
#     # if not tf_window.empty and 'swap_qty' in tf_window.columns:
#     #     print(f"tf_window columns: {tf_window.columns}")
#     #     features['swap_trade_volume_2min'] = format_float(tf_window['swap_qty'].fillna(0).sum())
#     #     features['swap_trade_count_2min'] = len(tf_window)
#     #     if 'trade_type' in tf_window.columns:
#     #         buy_trades = (tf_window['trade_type'] == 'BUY').sum()
#     #         features['swap_buy_trade_ratio'] = format_float(buy_trades / len(tf_window))
#     # else:
#     #     features['swap_trade_volume_2min'] = 0
#     #     features['swap_trade_count_2min'] = 0
#     #     features['swap_buy_trade_ratio'] = np.nan


#     # === Execution Context Features ===
#     # These come from trade_record, added for model context
#     features['execute_delay_ms'] = trade_record.get('execute_delay_ms', np.nan)
#     features['threshold'] = trade_record.get('threshold', np.nan)
#     features['basis_expected'] = trade_record.get('basis_expected', np.nan)
#     features['basis_executed'] = trade_record.get('basis_executed', np.nan)
    
#     # Slippage in ticks
#     basis_slippage = trade_record.get('basis_slippage', np.nan)
#     if not pd.isna(basis_slippage) and avg_tick > 0:
#         features['basis_slippage_ticks'] = format_float(basis_slippage / avg_tick)
#     else:
#         features['basis_slippage_ticks'] = np.nan
    
#     return features
def extract_window_features(
    ob_window: pd.DataFrame,
    tf_window: pd.DataFrame,
    spot_tick: float,
    swap_tick: float,
    trade_record: Dict
) -> Dict[str, float]:
    """
    Extract features from 60-second market data window.
    修复了空窗口/单样本导致的 RuntimeWarning 问题。
    """
    features = {}

    if ob_window.empty:
        feature_list = [
            'basis_ask_mean', 'basis_bid_mean', 'basis_ask_std', 'basis_bid_std', 
            'basis_ask_open', 'basis_bid_open', 'basis_ask_close', 'basis_bid_close', 
            'basis_ask_high', 'basis_bid_high', 'basis_ask_low', 'basis_bid_low',
            'basis_ask_adjusted_mean', 'basis_bid_adjusted_mean', 'basis_ask_adjusted_std', 'basis_bid_adjusted_std',
            'spot_spread_ticks', 'spot_spread_mean', 'swap_spread_ticks', 'swap_spread_mean', 
            'spot_depth_imbalance_mean', 'swap_depth_imbalance_mean', 'spot_depth5_imbalance_mean', 'swap_depth5_imbalance_mean',
            'spot_depth1_bid_ticks', 'spot_depth1_ask_ticks', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks',
            'spot_volatility_ticks', 'swap_volatility_ticks', 'spot_price_return_2min', 'swap_price_return_2min',
            'spot_trade_volume_2min', 'spot_trade_count_2min', 'spot_bid_price_mean', 'spot_ask_price_mean',
            'swap_bid_price_mean', 'swap_ask_price_mean', 'swap_trade_volume_2min', 'swap_trade_count_2min',
            'spot_buy_trade_ratio', 'execute_delay_ms', 'threshold', 'basis_expected', 'basis_executed', 'basis_slippage_ticks',
            'spot_midprice_mean', 'spot_midprice_std', 'spread_ticks', 
            'depth_imbalance_mean', 'volatility_ticks', 'trade_volume_2min',
            'spot_midprice_open', 'spot_midprice_close', 'spot_midprice_high', 'spot_midprice_low',
            'swap_midprice_mean', 'swap_midprice_std', 'swap_spread_mean', 
            'swap_depth_imbalance_mean', 'swap_depth1_bid_ticks', 'swap_depth1_ask_ticks', 'swap_buy_trade_ratio',
            # 兼容新增的加权特征占位
            'basis_mid_weighted_mean_-6', 'basis_mid_weighted_std_-6',
            'basis_mid_weighted_mean_-5', 'basis_mid_weighted_std_-5',
            'basis_mid_weighted_mean_-4', 'basis_mid_weighted_std_-4',
            'basis_mid_weighted_mean_-3', 'basis_mid_weighted_std_-3',
            'basis_mid_weighted_mean_-2', 'basis_mid_weighted_std_-2',
            'basis_mid_weighted_mean_-1', 'basis_mid_weighted_std_-1',
            'basis_mid_weighted_mean_0', 'basis_mid_weighted_std_0',
            'basis_mid_weighted_mean_1', 'basis_mid_weighted_std_1',
            'basis_mid_weighted_mean_2', 'basis_mid_weighted_std_2',
            'basis_mid_weighted_mean_3', 'basis_mid_weighted_std_3',
            'basis_mid_weighted_mean_4', 'basis_mid_weighted_std_4',
            'basis_mid_weighted_mean_5', 'basis_mid_weighted_std_5',
            'basis_mid_weighted_mean_6', 'basis_mid_weighted_std_6',
            'basis_mid_adjusted_mean', 'basis_mid_adjusted_std',
            'basis_mid_capped_mean', 'basis_mid_capped_std',
            'n_basis_bid_mix', 'n_basis_ask_mix', 'n_basis_capped',
        ]
        return {k: np.nan for k in feature_list}

    # 🛡️ 安全统计辅助函数：自动处理 NaN、空序列、单样本自由度问题
    def safe_stats(s):
        s = s.dropna()
        if len(s) == 0:
            return np.nan, np.nan
        m = s.mean()
        if len(s) < 2:
            return m, np.nan  # 标准差在样本<2时无意义，返回 NaN 避免警告
        return m, s.std()

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
    basis_ask_bid = np.log(swap_ask1) - np.log(spot_bid1)
    basis_bid_ask = np.log(swap_bid1) - np.log(spot_ask1)

    basis_ask_mix = np.concatenate([basis_ask.values.reshape(-1), basis_ask_bid.values.reshape(-1)])
    basis_bid_mix = np.concatenate([basis_bid.values.reshape(-1), basis_bid_ask.values.reshape(-1)])
    basis_mid = np.log(swap_midprice) - np.log(spot_midprice)

    # === Weighted basis adjustment (Softmax) ===
    weights = [-1e6, -1e5, -1e4, -1e3, -1e2, -10, -1, 0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    basis_mid_clean = basis_mid.dropna()
    
    if len(basis_mid_clean) == 0:
        for w in weights:
            # sign = np.sign(w)
            sign = '-' if w < 0 else ''
            log10_w = int(np.log10(np.abs(w))) if w != 0 else 0
            key_suffix = f'{sign}{log10_w}' if w != 0 else 'zero'
            features[f'basis_mid_weighted_mean_{key_suffix}'] = np.nan
            features[f'basis_mid_weighted_std_{key_suffix}'] = np.nan
    else:
        for w in weights:
            # 🛡️ 防止 exp 溢出导致 RuntimeWarning
            scaled = np.clip(w * basis_mid_clean, -500, 500)
            exp_vals = np.exp(scaled)
            denom = np.sum(exp_vals)
            if denom == 0 or np.isnan(denom):
                wght = np.zeros_like(basis_mid_clean)
            else:
                wght = exp_vals / denom
            basis_mid_weighted = basis_mid_clean * wght
            
            if w != 0:
                # sign = np.sign(w)
                sign = '-' if w < 0 else ''
                log10_w = int(np.log10(np.abs(w)))
                key_suffix = f'{sign}{log10_w}'
                features[f'basis_mid_weighted_mean_{key_suffix}'] = format_float(np.mean(basis_mid_weighted))
                features[f'basis_mid_weighted_std_{key_suffix}'] = format_float(np.std(basis_mid_weighted) if len(basis_mid_weighted) > 1 else np.nan)
            else:
                features['basis_mid_weighted_mean_zero'] = format_float(np.mean(basis_mid_weighted))
                features['basis_mid_weighted_std_zero'] = format_float(np.std(basis_mid_weighted) if len(basis_mid_weighted) > 1 else np.nan)

    # 使用 safe_stats 替换直接调用 .mean()/.std()
    m, s = safe_stats(spot_midprice)
    features['spot_midprice_mean'] = format_float(m)
    features['spot_midprice_std'] = format_float(s)

    m, _ = safe_stats(spot_spread)
    features['spot_spread_mean'] = format_float(m)

    m, s = safe_stats(swap_midprice)
    features['swap_midprice_mean'] = format_float(m)
    features['swap_midprice_std'] = format_float(s)

    m, _ = safe_stats(swap_spread)
    features['swap_spread_mean'] = format_float(m)

    features['spot_midprice_open'] = format_float(spot_midprice.iloc[0])
    features['spot_midprice_close'] = format_float(spot_midprice.iloc[-1])
    features['spot_midprice_high'] = format_float(spot_midprice.max())
    features['spot_midprice_low'] = format_float(spot_midprice.min())

    avg_tick = (spot_tick + swap_tick) / 2
    features['spot_spread_ticks'] = format_float((spot_spread / spot_tick).mean()) if spot_tick > 0 else np.nan
    features['swap_spread_ticks'] = format_float((swap_spread / swap_tick).mean()) if swap_tick > 0 else np.nan

    m, s = safe_stats(basis_ask)
    features['basis_ask_mean'] = format_float(m)
    features['basis_ask_std'] = format_float(s)
    features['basis_ask_open'] = format_float(basis_ask.iloc[0])
    features['basis_ask_close'] = format_float(basis_ask.iloc[-1])
    features['basis_ask_high'] = format_float(basis_ask.max())
    features['basis_ask_low'] = format_float(basis_ask.min())

    m, s = safe_stats(basis_bid)
    features['basis_bid_mean'] = format_float(m)
    features['basis_bid_std'] = format_float(s)
    features['basis_bid_open'] = format_float(basis_bid.iloc[0])
    features['basis_bid_close'] = format_float(basis_bid.iloc[-1])
    features['basis_bid_high'] = format_float(basis_bid.max())
    features['basis_bid_low'] = format_float(basis_bid.min())

    # === Basis Capped Logic ===
    if trade_record.get('mode') == 2:
        use_bid = trade_record.get('operation') == 'open2'
    else:
        use_bid = trade_record.get('operation') != 'open2'

    series = basis_bid_mix if use_bid else basis_ask_mix
    percentile = 90 if use_bid else 10
    bound = np.percentile(series, percentile)
    basis_capped = series[series < bound] if use_bid else series[series > bound]

    features['n_basis_bid_mix'] = len(basis_bid_mix)
    features['n_basis_ask_mix'] = len(basis_ask_mix)
    features['n_basis_capped'] = len(basis_capped)
    if len(basis_capped) == 0:
        features['basis_mid_adjusted_mean'] = format_float(np.mean(series))
        features['basis_mid_adjusted_std'] = format_float(np.std(series, ddof=1) if len(series) > 1 else np.nan)
        features['basis_mid_capped_mean'] = np.nan
        features['basis_mid_capped_std'] = np.nan
    else:
        features['basis_mid_adjusted_mean'] = format_float(np.mean(basis_capped))
        features['basis_mid_adjusted_std'] = format_float(np.std(basis_capped, ddof=1) if len(basis_capped) > 1 else np.nan)
        features['basis_mid_capped_mean'] = format_float(np.mean(basis_capped))
        features['basis_mid_capped_std'] = format_float(np.std(basis_capped, ddof=1) if len(basis_capped) > 1 else np.nan)

    # === Liquidity Features ===
    spot_bid1_qty = ob_window['spot_bid1_qty']
    spot_ask1_qty = ob_window['spot_ask1_qty']
    swap_bid1_qty = ob_window['swap_bid1_qty']
    swap_ask1_qty = ob_window['swap_ask1_qty']
    
    spot_bid5_qty = ob_window['spot_bid1_qty'] + ob_window['spot_bid2_qty'] + ob_window['spot_bid3_qty'] + ob_window['spot_bid4_qty'] + ob_window['spot_bid5_qty']
    spot_ask5_qty = ob_window['spot_ask1_qty'] + ob_window['spot_ask2_qty'] + ob_window['spot_ask3_qty'] + ob_window['spot_ask4_qty'] + ob_window['spot_ask5_qty']
    swap_bid5_qty = ob_window['swap_bid1_qty'] + ob_window['swap_bid2_qty'] + ob_window['swap_bid3_qty'] + ob_window['swap_bid4_qty'] + ob_window['swap_bid5_qty']
    swap_ask5_qty = ob_window['swap_ask1_qty'] + ob_window['swap_ask2_qty'] + ob_window['swap_ask3_qty'] + ob_window['swap_ask4_qty'] + ob_window['swap_ask5_qty']

    spot_depth_imbalance = (spot_bid1_qty - spot_ask1_qty) / (spot_bid1_qty + spot_ask1_qty + 1e-10)
    m, _ = safe_stats(spot_depth_imbalance)
    features['spot_depth_imbalance_mean'] = format_float(m)

    swap_depth_imbalance = (swap_bid1_qty - swap_ask1_qty) / (swap_bid1_qty + swap_ask1_qty + 1e-10)
    m, _ = safe_stats(swap_depth_imbalance)
    features['swap_depth_imbalance_mean'] = format_float(m)

    spot_depth5_imbalance = (spot_bid5_qty - spot_ask5_qty) / (spot_bid5_qty + spot_ask5_qty + 1e-10)
    m, _ = safe_stats(spot_depth5_imbalance)
    features['spot_depth5_imbalance_mean'] = format_float(m)

    swap_depth5_imbalance = (swap_bid5_qty - swap_ask5_qty) / (swap_bid5_qty + swap_ask5_qty + 1e-10)
    m, _ = safe_stats(swap_depth5_imbalance)
    features['swap_depth5_imbalance_mean'] = format_float(m)

    # Top-of-book depth in tick units (安全除法)
    def safe_tick_div(qty, px, tick):
        if tick <= 0: return np.nan
        s = (qty * px / tick).dropna()
        return s.mean() if len(s) > 0 else np.nan

    features['spot_depth1_bid_ticks'] = format_float(safe_tick_div(spot_bid1_qty, spot_bid1, spot_tick))
    features['spot_depth1_ask_ticks'] = format_float(safe_tick_div(spot_ask1_qty, spot_ask1, spot_tick))
    features['swap_depth1_bid_ticks'] = format_float(safe_tick_div(swap_bid1_qty, swap_bid1, swap_tick))
    features['swap_depth1_ask_ticks'] = format_float(safe_tick_div(swap_ask1_qty, swap_ask1, swap_tick))

    # === Volatility Features ===
    _, s_spot = safe_stats(spot_midprice)
    features['spot_volatility_ticks'] = format_float(s_spot / avg_tick) if avg_tick > 0 and not pd.isna(s_spot) else np.nan

    _, s_swap = safe_stats(swap_midprice)
    features['swap_volatility_ticks'] = format_float(s_swap / avg_tick) if avg_tick > 0 and not pd.isna(s_swap) else np.nan

    # Price return over window (安全首尾取值)
    valid_spot = spot_midprice.dropna()
    if len(valid_spot) >= 2:
        features['spot_price_return_2min'] = format_float((valid_spot.iloc[-1] - valid_spot.iloc[0]) / (valid_spot.iloc[0] + 1e-10))
    else:
        features['spot_price_return_2min'] = np.nan

    valid_swap = swap_midprice.dropna()
    if len(valid_swap) >= 2:
        features['swap_price_return_2min'] = format_float((valid_swap.iloc[-1] - valid_swap.iloc[0]) / (valid_swap.iloc[0] + 1e-10))
    else:
        features['swap_price_return_2min'] = np.nan

    # === Trade Flow Features ===
    if not tf_window.empty:
        bid_qty_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'bid_qty'].fillna(0)
        bid_px_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'bid_px'].fillna(0)
        bid_px_qty_spot = bid_px_spot * bid_qty_spot
        ask_qty_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'ask_qty'].fillna(0)
        ask_px_spot = tf_window.loc[tf_window['trade_type'] == 'spot', 'ask_px'].fillna(0)
        ask_px_qty_spot = ask_px_spot * ask_qty_spot   
        
        bid_qty_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'bid_qty'].fillna(0)
        bid_px_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'bid_px'].fillna(0)
        bid_px_qty_swap = bid_px_swap * bid_qty_swap
        ask_qty_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'ask_qty'].fillna(0)
        ask_px_swap = tf_window.loc[tf_window['trade_type'] == 'swap', 'ask_px'].fillna(0)
        ask_px_qty_swap = ask_px_swap * ask_qty_swap
        
        features['spot_trade_volume_2min'] = format_float((bid_qty_spot + ask_qty_spot).sum())
        features['spot_bid_price_mean'] = format_float(bid_px_qty_spot.sum() / (bid_qty_spot.sum() + 1e-10)) if bid_qty_spot.sum() > 0 else np.nan
        features['spot_ask_price_mean'] = format_float(ask_px_qty_spot.sum() / (ask_qty_spot.sum() + 1e-10)) if ask_qty_spot.sum() > 0 else np.nan
        features['spot_trade_count_2min'] = len(tf_window.loc[tf_window['trade_type'] == 'spot'])
        features['spot_buy_trade_ratio'] = format_float(np.log(bid_px_qty_spot.sum() + 1e-10) - np.log(ask_px_qty_spot.sum() + 1e-10)) if ask_px_qty_spot.sum()*bid_px_qty_spot.sum() > 0 else np.nan
        
        features['swap_trade_volume_2min'] = format_float((bid_qty_swap + ask_qty_swap).sum())
        features['swap_trade_count_2min'] = len(tf_window.loc[tf_window['trade_type'] == 'swap'])
        features['swap_bid_price_mean'] = format_float(bid_px_qty_swap.sum() / (bid_qty_swap.sum() + 1e-10)) if bid_qty_swap.sum() > 0 else np.nan
        features['swap_ask_price_mean'] = format_float(ask_px_qty_swap.sum() / (ask_qty_swap.sum() + 1e-10)) if ask_qty_swap.sum() > 0 else np.nan
        features['swap_buy_trade_ratio'] = format_float(np.log(bid_px_qty_swap.sum() + 1e-10) - np.log(ask_px_qty_swap.sum() + 1e-10)) if bid_px_qty_swap.sum()*ask_px_qty_swap.sum() > 0 else np.nan

    # === Execution Context Features ===
    features['execute_delay_ms'] = trade_record.get('execute_delay_ms', np.nan)
    features['threshold'] = trade_record.get('threshold', np.nan)
    features['basis_expected'] = trade_record.get('basis_expected', np.nan)
    features['basis_executed'] = trade_record.get('basis_executed', np.nan)

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
    # print(f"Preparing features for trade record: {result}")
    # Preparing features for trade record: {'date': '2026-01-01 17:29:56', 'symbol': 'DIAUSDT', 
    #                                       'operation': 'open2', 'trade_mode': 2, 
    #                                       'maker_spot_client_order_id': 'so22000npeZMijXABf1SwxkMxYnJcGWv', 
    #                                       'taker_swap_client_order_id': 'fo22001npeZMijXABf1SwxkMxYnJcGWv', 
    #                                       'threshold': -0.0007853173272683, 'taker/swap_place_price': 0.2615, 
    #                                       'taker/swap_withdraw_price': 0.2616, 'maker_spot_anticipated_price': 0.2617, 
    #                                       'anticipated_basis': 0.0, 'maker_spot_executed_qty': 77.0, 'maker_spot_executed_price': 0.2617, 
    #                                       'taker_swap_executed_price': 0.2614, 'executed_basis': -0.0011463507833397, 
    #                                       'taker_swap_haircut_executed_qty': 77.0, 'executed_volume': 20.1278, 
    #                                       'haircut_volume': 0.0, 'gain_vs_threshold': -0.0003610334560713, 
    #                                       'timer_start_ts': 1767288562032, 'maker_spot_executed_ts': 1767288596126, 
    #                                       'taker_swap_haircut_executed_ts': 1767288596129, 'is_maker/spot_maker': True, 
    #                                       'is_taker/swap_maker': False, 'current_taker_price': '0.2616', 
    #                                       'anticipated_basis_secondary': '0', 'spot_ticker_event_time': 1767288562018043, 
    #                                       'swap_ticker_event_time': 1767288562031, 'spot_depth_event_time': 1767288562025295, 
    #                                       'swap_depth_event_time': 1767288561730, 'spot_ticker_trade_time': 0, 
    #                                       'swap_ticker_trade_time': 0, 'spot_depth_trade_time': 0, 'swap_depth_trade_time': 1767288561671, 
    #                                       'spot_ticker_update_id': 1457623673, 'swap_ticker_update_id': 9579043394850, 
    #                                       'spot_depth_update_id': 1457623674, 'swap_depth_update_id': 9579043372056, 
    #                                       'swap_ticker_first_receive_trade_time': 1767288562018750, 
    #                                       'spot_ticker_first_receive_trade_time': 1767288562032432, 
    #                                       'spot_depth_first_receive_trade_time': 1767288562025686, 
    #                                       'swap_depth_first_receive_trade_time': 1767288561732044, 
    #                                       'swap_ticker_local_check_time': 1767288562018751, 
    #                                       'spot_ticker_local_check_time': 1767288562032434, 
    #                                       'spot_depth_local_check_time': 1767288562025688, 
    #                                       'swap_depth_local_check_time': 1767288561732054, 
    #                                       'trigger_type': 'SwapTicker', 'maker_level': -1, 
    #                                       'date_parsed': Timestamp('2026-01-01 17:29:56')}
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