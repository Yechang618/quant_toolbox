#!/usr/bin/env python3
"""
step1_extract_windows.py
第一步：提取交易记录及对应的1s重采样订单簿/交易流窗口，按symbol保存为Parquet。
✅ 已修复 Timestamp 序列化问题（Parquet 原生支持）
✅ 新增 max_symbols / max_records / debug 调试参数
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocess import load_trade_records, filter_valid_records, load_market_data
from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)

def extract_and_save_windows(
    trade_file_pattern: str = "combined_*.csv",
    date_start: str = '2026-01-01',
    date_end: str = '2026-01-31',
    symbols: list = None,
    output_dir: Path = None,
    window_sec: int = 60,
    max_symbols: int = None,
    max_records_per_symbol: int = None,
    debug: bool = False
):
    logger.info("=== Step 1: Extracting windows & saving to Parquet ===")
    if output_dir is None:
        output_dir = settings.output_root / "step1_windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载并过滤交易记录
    trade_df = load_trade_records(trade_file_pattern)
    trade_df = filter_valid_records(trade_df, date_start=date_start, date_end=date_end)

    unique_symbols = trade_df['symbol'].unique().tolist()
    unique_dates = trade_df['date_parsed'].dt.strftime('%Y%m%d').dropna().unique().tolist()
    if symbols:
        unique_symbols = [s for s in unique_symbols if s in symbols]
    if max_symbols:
        unique_symbols = unique_symbols[:max_symbols]
        logger.info(f"[DEBUG] 限制处理前 {max_symbols} 个 symbols: {unique_symbols}")

    logger.info(f"Processing {len(unique_symbols)} symbols across {len(unique_dates)} dates")

    for sym_idx, symbol in enumerate(unique_symbols, 1):
        if symbol in ['DIAUSDT', 'PROMUSDT']:
            logger.info(f"⚠️ Skipping symbol {symbol} due to known data issues")
            continue
        logger.info(f"[{sym_idx}/{len(unique_symbols)}] Extracting windows for symbol: {symbol}")
        symbol_data = []

        for date in unique_dates:
            ob_df = load_market_data(symbol, date, 'book')
            tf_df = load_market_data(symbol, date, 'trades')
            if ob_df.empty and tf_df.empty:
                continue

            # 1s 重采样（与原逻辑严格对齐）
            for df in (ob_df, tf_df):
                df.index = pd.to_datetime(df['time_str'], format='mixed', utc=True)
                df = df.sort_index()
                df = df[~df.index.duplicated(keep='first')]
                df = df.resample('s').ffill(limit=120)
                df.drop(columns=['time_str'], inplace=True, errors='ignore')
                df['ts_ms'] = (df.index.astype('int64') // 10**6).astype('int64')

            mask = (trade_df['symbol'] == symbol) & (trade_df['date_parsed'].dt.strftime('%Y%m%d') == date)
            records = trade_df[mask]
            if max_records_per_symbol:
                records = records.head(max_records_per_symbol)
                if debug: logger.debug(f"  [DEBUG] 截取前 {max_records_per_symbol} 条记录")

            for _, record in records.iterrows():
                exec_ts = record.get('taker_swap_haircut_executed_ts')
                if pd.isna(exec_ts):
                    continue

                start_ms = int(exec_ts - window_sec * 1000)
                end_ms = int(exec_ts + window_sec * 1000)

                ob_w = ob_df[(ob_df['ts_ms'] >= start_ms) & (ob_df['ts_ms'] <= end_ms)]
                tf_w = tf_df[(tf_df['ts_ms'] >= start_ms) & (tf_df['ts_ms'] <= end_ms)]

                # 清理冗余列，Parquet 会自动处理 NaN -> null
                rec_clean = record.drop(columns=['date_parsed', 'year_month'], errors='ignore')
                
                item = {
                    'record': rec_clean.where(pd.notna(rec_clean), None).to_dict(),
                    'ob_window': ob_w.where(pd.notna(ob_w), None).to_dict(orient='records'),
                    'tf_window': tf_w.where(pd.notna(tf_w), None).to_dict(orient='records')
                }
                
                if debug:
                    logger.debug(f"  [DEBUG] exec_ts={exec_ts}, ob_rows={len(ob_w)}, tf_rows={len(tf_w)}")
                symbol_data.append(item)

        # ✅ 修复原语法错误，并使用 Parquet 保存
        if symbol_data:
            out_path = output_dir / f"{symbol}_windows.parquet"
            df_out = pd.DataFrame(symbol_data)
            # pyarrow 自动处理 list[struct] 嵌套类型，无需自定义 Encoder
            df_out.to_parquet(out_path, engine='pyarrow', index=False)
            logger.info(f"✓ Saved {len(symbol_data)} windows for {symbol} → {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
        else:
            logger.warning(f"No valid windows extracted for {symbol}")

    logger.info("=== Step 1 Completed ===")

if __name__ == "__main__":
    extract_and_save_windows(
        trade_file_pattern="combined_*.csv",
        date_start='2026-01-01',
        date_end='2026-01-31',
        output_dir=Path("data/step1_windows"),
        max_symbols=None,               # 🐛 Debug 开关：默认只跑2个币种
        max_records_per_symbol=50,   # 🐛 Debug 开关：每个币种只取50条
        debug = False
    )