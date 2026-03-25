"""
data/preprocess.py
Main data preprocessing pipeline coordinator.
Modified to save results per symbol immediately after processing.
"""
import pandas as pd
import numpy as np
import gzip
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime

from util.config import settings
from util.helpers import (
    parse_timestamp, normalize_columns, extract_symbol_from_path, 
    ensure_dir, format_float
)
from util.logger import get_logger
from data.feature_engineering import (
    extract_window_features, prepare_trade_record_features
)
from util.binance_meta import get_ticksize_pair

logger = get_logger(__name__)


def load_trade_records(file_pattern: str) -> pd.DataFrame:
    """
    Load trade records from combined CSV files.
    
    Args:
        file_pattern: Glob pattern for trade record files
    
    Returns:
        DataFrame with normalized column names
    """
    files = list(Path(settings.bn_trade_root).glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No trade record files found matching: {file_pattern}")
    
    logger.info(f"Loading {len(files)} trade record file(s)")
    
    dfs = []
    for f in files:
        logger.debug(f"Reading {f}")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = normalize_columns(combined)
    
    # Parse timestamp columns
    if 'date' in combined.columns:
        combined['date_parsed'] = pd.to_datetime(combined['date'], errors='coerce')
    
    logger.info(f"Loaded {len(combined)} trade records")
    return combined


def filter_valid_records(
    df: pd.DataFrame, 
    date_start: Optional[str] = None, 
    date_end: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter trade records to valid modes, required fields, and date range.
    
    Args:
        df: Raw trade records DataFrame
        date_start: Start date in 'YYYY-MM' format (e.g., '2026-01')
        date_end: End date in 'YYYY-MM' format (e.g., '2026-01')
    
    Returns:
        Filtered DataFrame
    """
    logger.info(f"Initial records: {len(df)}")
    logger.info(f"Initial symbols: {df['symbol'].nunique()} unique")
    
    # Filter by date range (e.g., 2026-01)
    if date_start or date_end:
        if 'date_parsed' not in df.columns:
            logger.warning("date_parsed column not found, skipping date filter")
        else:
            # Create year-month column for filtering
            df['year_month'] = df['date_parsed'].dt.strftime('%Y-%m')
            
            if date_start and date_end:
                mask = (df['year_month'] >= date_start) & (df['year_month'] <= date_end)
                logger.info(f"Filtering date range: {date_start} to {date_end}")
            elif date_start:
                mask = df['year_month'] >= date_start
                logger.info(f"Filtering from date: {date_start}")
            elif date_end:
                mask = df['year_month'] <= date_end
                logger.info(f"Filtering until date: {date_end}")
            
            df = df[mask].copy()
            logger.info(f"Records after date filter: {len(df)}")
            
            # Clean up temporary column
            if 'year_month' in df.columns:
                df = df.drop(columns=['year_month'])
    
    # Filter by trade_mode
    valid_modes = [0, 2]
    df_filtered = df[df['trade_mode'].isin(valid_modes)].copy()
    logger.info(f"Records after mode filter (modes {valid_modes}): {len(df_filtered)}")
    
    # Filter by required fields
    required_fields = ['gain_vs_threshold', 'symbol', 'taker_swap_haircut_executed_ts']
    df_filtered = df_filtered.dropna(subset=required_fields)
    logger.info(f"Records after required fields filter: {len(df_filtered)}")
    
    logger.info(f"Filtered to {len(df_filtered)} valid records")
    logger.info(f"Unique symbols: {df_filtered['symbol'].nunique()}")
    
    if 'date_parsed' in df_filtered.columns:
        date_range = df_filtered['date_parsed'].agg(['min', 'max'])
        logger.info(f"Date range: {date_range['min']} to {date_range['max']}")
    
    return df_filtered


def load_market_data(
    symbol: str, 
    date: str, 
    data_type: str = 'book'
) -> pd.DataFrame:
    """
    Load orderbook or trade flow data for a specific symbol and date.
    
    Args:
        symbol: Trading pair symbol (may or may not include USDT suffix)
        date: Date string in 'YYYYMMDD' format
        data_type: 'book' for orderbook, 'trades' for trade flow
    
    Returns:
        DataFrame with parsed timestamps
    """
    # Handle symbol that may already include USDT
    symbol_upper = symbol.upper().strip()
    if symbol_upper.endswith('USDT'):
        symbol_base = symbol_upper[:-4]
        symbol_with_usdt = symbol_upper
    else:
        symbol_base = symbol_upper
        symbol_with_usdt = f"{symbol_upper}USDT"
    
    if data_type == 'book':
        filename = f"book_{symbol_with_usdt}_{date}.csv.gz"
    elif data_type == 'trades':
        filename = f"trades_{symbol_with_usdt}_{date}.csv.gz"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Build path with symbol directory (includes USDT)
    filepath = settings.market_processed_root / date / symbol_with_usdt / filename
    
    if not filepath.exists():
        return pd.DataFrame()
    
    logger.debug(f"Loading {data_type} data: {filepath}")
    
    # Read gzipped CSV
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)
    
    # Parse time_str to timestamp
    if 'time_str' in df.columns:
        df['ts_ms'] = df['time_str'].apply(
            lambda x: int(parse_timestamp(x).timestamp() * 1000) if pd.notna(x) else np.nan
        )
    
    return df


def process_single_record(
    record: pd.Series,
    ob_df: pd.DataFrame,
    tf_df: pd.DataFrame,
    window_sec: int = 60
) -> Optional[Dict]:
    """
    Process a single trade record: extract window and generate features.
    
    Args:
        record: Single trade record
        ob_df: Full orderbook DataFrame for the symbol/date
        tf_df: Full trade flow DataFrame for the symbol/date
        window_sec: Backtrack window in seconds
    
    Returns:
        Dict with features + labels, or None if processing failed
    """
    try:
        # Prepare record with derived fields
        record_dict = prepare_trade_record_features(record)
        
        # Get execution timestamp (ms)
        exec_ts = record_dict.get('taker_swap_haircut_executed_ts')
        if pd.isna(exec_ts):
            logger.warning(f"Missing execution timestamp for record")
            return None
        
        # Define time window
        window_start_ms = exec_ts - window_sec * 1000
        
        # Filter market data to window
        ob_window = ob_df[
            (ob_df['ts_ms'] >= window_start_ms) & 
            (ob_df['ts_ms'] <= exec_ts)
        ].copy() if not ob_df.empty else pd.DataFrame()
        
        tf_window = tf_df[
            (tf_df['ts_ms'] >= window_start_ms) & 
            (tf_df['ts_ms'] <= exec_ts)
        ].copy() if not tf_df.empty else pd.DataFrame()
        
        # Get ticksize
        symbol = record_dict.get('symbol', '')
        spot_tick, swap_tick = get_ticksize_pair(symbol)
        
        # Extract features
        features = extract_window_features(
            ob_window, tf_window, spot_tick, swap_tick, record_dict
        )
        
        # Combine with labels and metadata
        result = {
            # Labels
            'gain_vs_threshold': record_dict.get('gain_vs_threshold'),
            'basis_slippage': record_dict.get('basis_slippage'),
            
            # Meta
            'symbol': symbol,
            'trade_mode': record_dict.get('trade_mode'),
            'exec_ts_utc': parse_timestamp(exec_ts),
            
            # Time fields
            'execute_delay_ms': record_dict.get('execute_delay_ms'),
            'timer_start_ts': record_dict.get('timer_start_ts'),
            'taker_exec_ts': exec_ts,
            
            # Price fields
            'threshold': record_dict.get('threshold'),
            'basis_expected': record_dict.get('basis_expected'),
            'basis_executed': record_dict.get('basis_executed'),
            
            # Features
            **features
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing record: {e}", exc_info=True)
        return None


def save_symbol_results(
    results: Dict[int, List[Dict]], 
    symbol: str, 
    dry_run: bool = False
):
    """
    Save processed results for a single symbol to CSV files.
    
    Args:
        results: Dictionary of mode -> list of result dicts
        symbol: Trading pair symbol
        dry_run: If True, only log without saving
    """
    for mode in [0, 2]:
        mode_results = results.get(mode, [])
        if not mode_results:
            logger.debug(f"No results for symbol={symbol}, mode={mode}")
            continue
        
        # Convert to DataFrame
        df_out = pd.DataFrame(mode_results)
        
        # Determine output path
        output_path = (
            settings.output_mode0 if mode == 0 else settings.output_mode2
        ) / f"sample_{symbol}.csv"
        
        if dry_run:
            logger.info(f"[DRY RUN] Would save {len(df_out)} rows to {output_path}")
        else:
            # Ensure directory exists
            ensure_dir(str(output_path.parent))
            
            # Save to CSV
            df_out.to_csv(output_path, index=False)
            print(f"Symbol={symbol}, columns={df_out.columns.tolist()}")
            print(df_out.head())
            logger.info(f"✓ Saved {len(df_out)} rows for symbol={symbol}, mode={mode} → {output_path}")


def run_pipeline(
    trade_file_pattern: str = "combined_*.csv",
    symbols: Optional[List[str]] = None,
    dates: Optional[List[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    dry_run: bool = False
):
    """
    Run the full preprocessing pipeline.
    
    Args:
        trade_file_pattern: Pattern for trade record files
        symbols: Optional list of symbols to process (None = all)
        dates: Optional list of dates to process in YYYYMMDD format (None = all found in records)
        date_start: Optional start date in YYYY-MM format (e.g., '2026-01')
        date_end: Optional end date in YYYY-MM format (e.g., '2026-01')
        dry_run: If True, only log actions without saving output
    """
    logger.info("=== Starting preprocessing pipeline ===")
    
    # Load and filter trade records
    trade_df = load_trade_records(trade_file_pattern)
    trade_df = filter_valid_records(trade_df, date_start=date_start, date_end=date_end)
    
    # Determine unique symbols and dates
    unique_symbols = trade_df['symbol'].unique().tolist()
    unique_dates = trade_df['date_parsed'].dt.strftime('%Y%m%d').dropna().unique().tolist()
    
    if symbols:
        unique_symbols = [s for s in unique_symbols if s in symbols]
    if dates:
        unique_dates = [d for d in unique_dates if d in dates]
    
    logger.info(f"Processing {len(unique_symbols)} symbols across {len(unique_dates)} dates")
    
    # Prepare output directories
    if not dry_run:
        ensure_dir(settings.output_mode0)
        ensure_dir(settings.output_mode2)
    
    # Track overall statistics
    total_records_processed = 0
    total_records_saved = 0
    symbols_completed = 0
    
    # === MODIFIED: Process and save per symbol ===
    for symbol_idx, symbol in enumerate(unique_symbols, 1):
        logger.info(f"[{symbol_idx}/{len(unique_symbols)}] Processing symbol: {symbol}")
        
        # Initialize results for this symbol
        symbol_results = {0: [], 2: []}
        
        for date in unique_dates:
            # Load market data for this symbol+date
            ob_df = load_market_data(symbol, date, 'book')
            tf_df = load_market_data(symbol, date, 'trades')
            
            if ob_df.empty and tf_df.empty:
                logger.debug(f"No market data for {symbol} on {date}")
                continue
            
            logger.debug(f"Loaded market data for {symbol} on {date}: "
                        f"{len(ob_df)} orderbook rows, {len(tf_df)} trade rows")
            
            # Filter trade records for this symbol+date
            mask = (
                (trade_df['symbol'] == symbol) & 
                (trade_df['date_parsed'].dt.strftime('%Y%m%d') == date)
            )
            records = trade_df[mask]
            
            if records.empty:
                continue
            
            logger.debug(f"Processing {len(records)} records for {symbol}/{date}")
            
            # Process each record
            for _, record in records.iterrows():
                result = process_single_record(record, ob_df, tf_df)
                if result:
                    mode = result['trade_mode']
                    if mode in symbol_results:
                        symbol_results[mode].append(result)
                        total_records_processed += 1
        
        # === Save results for this symbol immediately ===
        symbol_total = sum(len(symbol_results[mode]) for mode in [0, 2])
        if symbol_total > 0:
            save_symbol_results(symbol_results, symbol, dry_run)
            total_records_saved += symbol_total
            symbols_completed += 1
            logger.info(f"[{symbol_idx}/{len(unique_symbols)}] Completed symbol={symbol}: "
                       f"{symbol_total} records saved")
        else:
            logger.warning(f"No valid results for symbol={symbol}")
        
        # Optional: Clear memory for this symbol
        del symbol_results
    # ==============================================
    
    # Final summary
    logger.info("=== Pipeline completed ===")
    logger.info(f"Total symbols processed: {symbols_completed}/{len(unique_symbols)}")
    logger.info(f"Total records processed: {total_records_processed}")
    logger.info(f"Total records saved: {total_records_saved}")
    if not dry_run:
        logger.info(f"Output directories:")
        logger.info(f"  Mode 0: {settings.output_mode0}")
        logger.info(f"  Mode 2: {settings.output_mode2}")


if __name__ == "__main__":
    # Example usage - Process only January 2026 data
    run_pipeline(
        date_start='2026-01',
        date_end='2026-01',
        dry_run=False
    )