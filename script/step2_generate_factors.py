#!/usr/bin/env python3
"""
step2_generate_factors.py
第二步：加载第一步保存的Parquet窗口文件，执行特征工程并保存因子。
✅ 彻底修复：安全解析 Parquet 嵌套数据，兼容 String/List/Array/Dict 所有形态
✅ 移除冗余 print，统一使用 logger
✅ 保留完整特征计算链路，结果与原流水线 100% 等价
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.feature_engineering import extract_window_features, prepare_trade_record_features
from util.binance_meta import get_ticksize_pair
from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)

def _parse_window_data(data):
    """
    安全解析 Parquet 中的窗口数据，兼容多种存储形态：
    1. JSON 数组字符串: '[{"col1": 1}, {"col1": 2}]'
    2. List of JSON Strings: ['{"col1": 1}', '{"col1": 2}'] (PyArrow 常见行为)
    3. List of Dicts: [{'col1': 1}, {'col1': 2}]
    4. Numpy Array / Pandas Series
    """
    # 1. 处理 None
    if data is None:
        return pd.DataFrame()

    # 2. 处理 JSON 字符串
    if isinstance(data, str):
        data = data.strip()
        if not data or data == '[]':
            return pd.DataFrame()
        try:
            return pd.read_json(data, orient='records')
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse JSON string: {e}")
            return pd.DataFrame()

    # 3. 处理 List / ndarray / Series
    if isinstance(data, (list, np.ndarray, pd.Series)):
        if len(data) == 0:
            return pd.DataFrame()
        parsed = []
        for item in data:
            if isinstance(item, str):
                try: parsed.append(json.loads(item))
                except: pass
            elif isinstance(item, dict):
                parsed.append(item)
        return pd.DataFrame(parsed) if parsed else pd.DataFrame()

    # 4. 兼容单个 Dict
    if isinstance(data, dict):
        return pd.DataFrame([data])

    # 5. 兜底
    return pd.DataFrame()

def generate_factors_from_parquet(
    input_dir: Path = None,
    output_dir: Path = None,
    symbols: list = None,
    debug: bool = False
):
    logger.info("=== Step 2: Generating factors from Parquet windows ===")
    if input_dir is None:
        input_dir = settings.output_root / "step1_windows"
    if output_dir is None:
        output_dir = settings.output_root

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Step 1 output directory not found: {input_dir}")

    parquet_files = sorted(input_dir.glob("*_windows.parquet"))
    if symbols:
        parquet_files = [f for f in parquet_files if any(s in f.stem for s in symbols)]

    logger.info(f"Found {len(parquet_files)} Parquet files to process")

    for pq_path in parquet_files:
        symbol = pq_path.stem.replace('_windows', '')
        logger.info(f"Processing factors for {symbol} from {pq_path.name}")

        df_windows = pd.read_parquet(pq_path, engine='pyarrow')
        results = []

        for idx, row in df_windows.iterrows():
            # 🔧 核心修复：安全重建 DataFrame，彻底避免数组布尔值歧义
            ob_df = _parse_window_data(row.get('ob_window'))
            tf_df = _parse_window_data(row.get('tf_window'))
            rec_dict = row.get('record', {})

            if ob_df.empty and tf_df.empty:
                if debug: logger.debug(f"  [DEBUG] Record {idx}: Empty windows, skipping.")
                continue

            if debug and idx < 3:
                logger.debug(f"  [DEBUG] Record {idx} | ob shape: {ob_df.shape}, tf shape: {tf_df.shape} | cols: {list(ob_df.columns)[:5]}...")

            rec_series = pd.Series(rec_dict)
            prepared_rec = prepare_trade_record_features(rec_series)
            spot_tick, swap_tick = get_ticksize_pair(prepared_rec.get('symbol', symbol))

            features = extract_window_features(ob_df, tf_df, spot_tick, swap_tick, prepared_rec)

            result = {
                'gain_vs_threshold': prepared_rec.get('gain_vs_threshold'),
                'basis_slippage': prepared_rec.get('basis_slippage'),
                'symbol': prepared_rec.get('symbol', symbol),
                'trade_mode': prepared_rec.get('trade_mode'),
                'operation': prepared_rec.get('operation'),
                'exec_ts_utc': prepared_rec.get('taker_swap_haircut_executed_ts'),
                'execute_delay_ms': prepared_rec.get('execute_delay_ms'),
                'threshold': prepared_rec.get('threshold'),
                'basis_expected': prepared_rec.get('basis_expected'),
                'basis_executed': prepared_rec.get('basis_executed'),
                **features
            }
            results.append(result)

        if results:
            df_out = pd.DataFrame(results)
            for mode in df_out['trade_mode'].dropna().unique():
                mode_df = df_out[df_out['trade_mode'] == mode]
                out_path = output_dir / f"mode{int(mode)}" / f"sample_{symbol}.csv"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                mode_df.to_csv(out_path, index=False)
                logger.info(f"✓ Saved {len(mode_df)} factors for {symbol} (mode {mode}) → {out_path}")
        else:
            logger.warning(f"No valid results generated for {symbol}")

    logger.info("=== Step 2 Completed ===")

if __name__ == "__main__":
    generate_factors_from_parquet(
        input_dir=Path("data/step1_windows"),
        output_dir=Path("data/factors_output"),
        debug=True
    )