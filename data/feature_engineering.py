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

def single_mixture_weighted_stats(series, weights):
    if len(series) == 0:
        return {f'{w}': (np.nan, np.nan) for w in weights}
    
    results = {}
    for w in weights:
        sign = '-' if w < 0 else ''
        log10_w = int(np.log10(np.abs(w))) if w != 0 else 0
        key_suffix = f'1e{sign}{log10_w}' if w != 0 else '0'
        scaled = np.clip(w * series, -500, 500)
        exp_vals = np.exp(scaled)
        denom = np.sum(exp_vals)
        if denom == 0 or np.isnan(denom):
            wght = np.zeros_like(series)
        else:
            wght = exp_vals / denom
        weighted_series = series * wght
        m = np.sum(weighted_series) / (np.sum(wght) + 1e-10)  # 安全除法
        results[f'{key_suffix}'] = m
    return results

def dual_mixture_weighted_stats(series_1, series_2, weights, alphas=[1.0]):
    if len(series_1) == 0 or len(series_2) == 0:
        return {f'{w}{a}': np.nan for w in weights for a in alphas}
    
    results = {}
    for w in weights:
        sign = '-' if w < 0 else ''
        log10_w = int(np.log10(np.abs(w))) if w != 0 else 0
        w_suffix = f'1e{sign}{log10_w}' if w != 0 else '0'
        for a in alphas:
            scaled_1 = np.clip(w * series_1, -500, 500)
            scaled_2 = np.clip(w * series_2, -500, 500)
            exp_vals_1 = np.exp(scaled_1)
            exp_vals_2 = np.exp(scaled_2)
            denom_1 = np.sum(exp_vals_1)
            denom_2 = np.sum(exp_vals_2)
            if denom_1 == 0 or np.isnan(denom_1):
                wght_1 = np.zeros_like(series_1)
            else:
                wght_1 = exp_vals_1 / denom_1
            if denom_2 == 0 or np.isnan(denom_2):
                wght_2 = np.zeros_like(series_2)
            else:
                wght_2 = exp_vals_2 / denom_2
            weighted_series_1 = series_1 * wght_1
            weighted_series_2 = series_2 * wght_2
            m = (np.sum(weighted_series_1)+ np.sum(weighted_series_2) * a) / (np.sum(wght_1) + np.sum(wght_2) * a + 1e-10)  # 安全除法
            results[f'{w_suffix}_{a}'] = m
    return results

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
    # print(f"ob_window size: {len(ob_window)}")
    # print(ob_window.info())
    # print(f"ob_window size: {len(tf_window)}")    
    # print(tf_window.info())

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

    basis_ask_mix = np.concatenate([basis_ask.values.reshape(-1), basis_bid_ask.values.reshape(-1)])
    basis_bid_mix = np.concatenate([basis_bid.values.reshape(-1), basis_ask_bid.values.reshape(-1)])
    basis_mid = np.log(swap_midprice) - np.log(spot_midprice)

    # === Weighted basis adjustment (Softmax) ===
    weights = [-1e6, -1e5, -1e4, -1e3, -1e2, -10, -1, 0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
    alphas = [0.5, 0.75, 1.0]  # 双重加权的第二层权重
    basis_mid = basis_mid.dropna()
    if trade_record.get('mode') == 0:
        if trade_record.get('operation') == 'open2':
            basis_1 = basis_ask
            basis_2 = basis_bid_ask
        else:
            basis_1 = basis_bid
            basis_2 = basis_ask_bid
    else:
        if trade_record.get('operation') == 'open2':
            basis_1 = basis_bid
            basis_2 = basis_bid_ask
        else:
            basis_1 = basis_ask
            basis_2 = basis_ask_bid


    if len(basis_mid) == 0:
        for w in weights:
            # sign = np.sign(w)
            sign = '-' if w < 0 else ''
            log10_w = int(np.log10(np.abs(w))) if w != 0 else 0
            key_suffix = f'1e{sign}{log10_w}' if w != 0 else '0'
            features[f'basis_mid_weighted_mean_{key_suffix}'] = np.nan
            features[f'basis_mid_weighted_std_{key_suffix}'] = np.nan
            for a in alphas:
                features[f'basis_mid_weighted_mean_{key_suffix}_{a}'] = np.nan
                features[f'basis_mid_weighted_std_{key_suffix}_{a}'] = np.nan
    else:
        results_single = single_mixture_weighted_stats(basis_mid, weights)
        for key_suffix, m in results_single.items():
            features[f'basis_mid_weighted_mean_{key_suffix}'] = format_float(m)
        results_dual = dual_mixture_weighted_stats(basis_1, basis_2, weights, alphas)
        for key_suffix, m in results_dual.items():
            features[f'basis_mid_weighted_mean_{key_suffix}'] = format_float(m)

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
    # Mode 2, operation close => basis_ask_mix = (basis_ask, basis_bid_ask) 去掉末尾的10%

    if trade_record.get('mode') == 2:
        use_bid = trade_record.get('operation') == 'open2'
    else:
        use_bid = trade_record.get('operation') != 'open2'

    # series = np.concatenate([basis_1, basis_2])
    # print(f"basis_1 shape: {basis_1.shape}, basis_2 shape: {basis_2.shape}, combined shape: {series.shape}")
    # if trade_record.get('operation') == 'open2':
    #     bound = np.percentile(series, 90) 
    #     basis_capped = series[series < bound]
    # else:
    #     bound = np.percentile(series, 10)
    #     basis_capped = series[series > bound]
    series = np.concatenate([basis_1, basis_2])
    if series.size == 0:
        raise ValueError("basis_1 和 basis_2 均为空，无法计算分位数")

    # 使用 nanpercentile 兼容 NaN
    bound = np.nanpercentile(series, 90 if trade_record.get('operation') == 'open2' else 10)

    if trade_record.get('operation') == 'open2':
        basis_capped = series[series <= bound]  # 改为 <= 更稳健
    else:
        basis_capped = series[series >= bound]

    features['n_basis_capped'] = len(basis_capped)
    if len(basis_capped) == 0:
        features['basis_mid_adjusted_mean'] = format_float(np.mean(series))
        features['basis_mid_adjusted_std'] = format_float(np.std(series, ddof=1) if len(series) > 1 else np.nan)
        features['basis_mid_capped_mean'] = np.nan
        features['basis_mid_capped_std'] = np.nan
    else:
        features['basis_mid_adjusted_mean'] = format_float(np.mean(series))
        features['basis_mid_adjusted_std'] = format_float(np.std(series, ddof=1) if len(basis_capped) > 1 else np.nan)
        features['basis_mid_capped_mean'] = format_float(np.mean(basis_capped))
        features['basis_mid_capped_std'] = format_float(np.std(basis_capped, ddof=1) if len(basis_capped) > 1 else np.nan)

    # Debug
    print(f"trade_record keys: {list(trade_record.keys())}")
    print(f"Extracted features for record with symbol={trade_record.get('symbol')}, exec_ts={trade_record.get('taker_swap_haircut_executed_ts')}")
    print(f"capped basis count: {features['n_basis_capped']}, capped mean: {features['basis_mid_capped_mean']}, capped std: {features['basis_mid_capped_std']}")
    print(f"weighted basis mean (w=1): {features.get('basis_mid_weighted_mean_1e0', np.nan)}")

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