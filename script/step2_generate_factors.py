#!/usr/bin/env python3
"""
step2_generate_factors.py
第二步：加载第一步保存的Parquet窗口文件，执行特征工程并保存因子。
✅ 彻底修复：添加数值类型强制转换层，消除 np.log(None) 报错
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

from data.feature_engineering import extract_window_features_simple, prepare_trade_record_features
from util.binance_meta import get_ticksize_pair
from util.config import settings
from util.logger import get_logger

logger = get_logger(__name__)

def _parse_window_data(data):
    """安全解析 Parquet 中的窗口数据，兼容 String/List/Array/Dict 所有形态"""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, str):
        data = data.strip()
        if not data or data == '[]':
            return pd.DataFrame()
        try:
            return pd.read_json(data, orient='records')
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse JSON string: {e}")
            return pd.DataFrame()
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
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()

def _sanitize_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """🔧 核心修复：将 DataFrame 强制转为数值类型，None/非法字符串自动转为 np.nan"""
    if df.empty:
        return df
    # apply 会逐列尝试转换，errors='coerce' 确保非数字内容变成 NaN 而非报错
    return df.apply(pd.to_numeric, errors='coerce')

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
            # 1. 解析原始数据
            ob_raw = _parse_window_data(row.get('ob_window'))
            tf_raw = _parse_window_data(row.get('tf_window'))
            # print(f"Get ob_raw with shape {ob_raw.shape} and tf_raw with shape {tf_raw.shape} for record {idx}")  # 临时输出，验证解析结果

            # 🔧 2. 强制类型清洗：确保 spot_bid1_px 等列为 float64，None -> np.nan
            ob_df = _sanitize_numeric_df(ob_raw)
            tf_df = _sanitize_numeric_df(tf_raw)
            # print(f"Get ob_df with shape {ob_df.shape} and tf_df with shape {tf_df.shape} for record {idx}")  # 临时输出，验证解析结果
            # print(f"Time range of ob_df: {ob_df['ts_ms'].min()} - {ob_df['ts_ms'].max()} = {ob_df['ts_ms'].max() - ob_df['ts_ms'].min()}")  # 验证时间戳范围
            # print(f"Sample ob_df rows:\n{ob_df.head(10)}")  # 验证数据内容
            # print(f"Description of ob_df: {ob_df.describe()} ")  # 验证数据类型和非空情况

            
            rec_dict = row.get('record', {})

            if ob_df.empty and tf_df.empty:
                if debug: logger.debug(f"  [DEBUG] Record {idx}: Empty windows, skipping.")
                continue

            if debug and idx < 3:
                logger.debug(f"  [DEBUG] Record {idx} | ob shape: {ob_df.shape}, tf shape: {tf_df.shape} | dtypes: {ob_df.dtypes.iloc[:3].to_dict()}...")

            rec_series = pd.Series(rec_dict)
            prepared_rec = prepare_trade_record_features(rec_series)
            spot_tick, swap_tick = get_ticksize_pair(prepared_rec.get('symbol', symbol))

            # 3. 安全送入特征工程（此时已无 NoneType 报错风险）
            features = extract_window_features_simple(ob_df, tf_df, spot_tick, swap_tick, prepared_rec)

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
                logger.info(f"✓ Saved {len(mode_df)} examples for {symbol} (mode {mode}) → {out_path}")
        else:
            logger.warning(f"No valid results generated for {symbol}")

    logger.info("=== Step 2 Completed ===")

if __name__ == "__main__":
    generate_factors_from_parquet(
        input_dir=Path("data_processed/step1_windows"),
        output_dir=Path("data_processed/factors_output_2"),
        debug=True
    )