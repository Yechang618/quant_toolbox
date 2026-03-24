import sys
import glob
import os
from pathlib import Path

# Ensure repo root is on sys.path so imports like `util` work when running
# the script directly (sys.path[0] is the script dir otherwise).
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import pandas as pd

print("START: data load checks")

workspace = Path.cwd()
# Search for trade files in several candidate locations (repo-relative and parent workspace)
trade_candidates = [
    workspace.joinpath('kronos_test','Kronos','dataset','bn_trade','combined_*.csv'),
    workspace.parent.joinpath('kronos_test','Kronos','dataset','bn_trade','combined_*.csv'),
    Path(r'C:\Users\yecha\workspace\kronos_test\Kronos\dataset\bn_trade\combined_*.csv')
]
matches = []
for tg in trade_candidates:
    found = glob.glob(str(tg))
    if found:
        matches = found
        break

if not matches:
    sample = workspace.joinpath('dataset','sample_trade.csv')
    if sample.exists():
        matches = [str(sample)]

if not matches:
    print("No trade files found under kronos_test/Kronos/dataset/bn_trade or sample_trade.csv")
    raise SystemExit(0)

trade_file = matches[0]
print(f"Found trade file: {trade_file}")

# Try to import normalize helper if available
try:
    from util.helpers import normalize_columns, parse_time_str_to_ms
    has_helpers = True
    print("Loaded util.helpers.normalize_columns and parse_time_str_to_ms")
except Exception as e:
    has_helpers = False
    print(f"Could not import util.helpers: {e}")

# Read a few rows from trade file
try:
    tf = pd.read_csv(trade_file, nrows=10)
    print("Trade file columns:")
    print(list(tf.columns))
    print("Dtypes:")
    print(tf.dtypes.to_dict())
    print("First rows:\n", tf.head(5).to_string(index=False))
except Exception as e:
    print(f"Failed to read trade file {trade_file}: {e}")
    raise

# Normalize column names if helper exists
if has_helpers:
    try:
        tf_norm = normalize_columns(tf.copy())
        print("Normalized trade columns:", list(tf_norm.columns))
    except Exception as e:
        print("normalize_columns failed:", e)

# Try to infer symbol and date from first row
symbol = None
date_str = None
cand_cols = ['symbol','pair','instId','instrument']
for c in cand_cols:
    if c in tf.columns:
        symbol = str(tf[c].iloc[0])
        break
# fallback to column names that may appear
if not symbol:
    for c in ['s','sym']:
        if c in tf.columns:
            symbol = str(tf[c].iloc[0])
            break

if 'date' in tf.columns:
    date_str = str(tf['date'].iloc[0])
else:
    # look for timestamp columns
    for c in ['ts','timestamp','timestamp_ms','timestamp_millis']:
        if c in tf.columns:
            date_str = str(tf[c].iloc[0])
            break

print(f"Inferred symbol: {symbol}")
print(f"Inferred date (raw): {date_str}")

# Prepare market file search. Try several candidate locations including
# the repo-relative path and the parent workspace path where kronos_test may live.
candidate_paths = [
    workspace.joinpath('kronos_test','Kronos','dataset','market_processed'),
    workspace.parent.joinpath('kronos_test','Kronos','dataset','market_processed'),
    Path(r'C:\Users\yecha\workspace\kronos_test\Kronos\dataset\market_processed')
]
market_root = None
for p in candidate_paths:
    if p.exists():
        market_root = p
        break

if market_root is None:
    print("Market root not found in candidate locations:")
    for p in candidate_paths:
        print(" -", p)
    raise SystemExit(0)
else:
    print(f"Using market root: {market_root}")

# If date looks like YYYY-MM-DD or YYYYMMDD, try to normalize
from datetime import datetime
market_date = None
if date_str:
    for fmt in ("%Y-%m-%d","%Y%m%d","%Y-%m-%d %H:%M:%S","%Y%m%d%H%M%S"):
        try:
            dt = datetime.strptime(date_str[:19], fmt)
            market_date = dt.strftime('%Y%m%d')
            break
        except Exception:
            pass

if not market_date:
    # try to parse numeric timestamp in ms
    try:
        ts = int(float(date_str))
        if ts > 1e12:
            dt = datetime.utcfromtimestamp(ts/1000)
        else:
            dt = datetime.utcfromtimestamp(ts)
        market_date = dt.strftime('%Y%m%d')
    except Exception:
        market_date = None

print(f"Normalized market date: {market_date}")

if not symbol or not market_date:
    print("Cannot form expected market file paths due to missing symbol/date")
    raise SystemExit(0)

# Expected folder structure: market_root/{date}/{symbol}USDT/
symbol_clean = symbol.replace('USDT','')
market_dir = market_root.joinpath(market_date, f"{symbol_clean}USDT")
print(f"Looking for market files in: {market_dir}")

if not market_dir.exists():
    print("Market directory not found")
    # try to list date folder
    date_dir = market_root.joinpath(market_date)
    if date_dir.exists():
        print("Date directory exists. Listing subfolders:")
        print(list(date_dir.iterdir())[:20])
    raise SystemExit(0)

# Search for book_ and trade (or trades_) files
book_glob = str(market_dir.joinpath(f"book_{symbol_clean}USDT_{market_date}*.csv*"))
trade_glob_patterns = [
    str(market_dir.joinpath(f"trade_{symbol_clean}USDT_{market_date}*.csv*")),
    str(market_dir.joinpath(f"trades_{symbol_clean}USDT_{market_date}*.csv*")),
]
book_matches = glob.glob(book_glob)
trade_matches = []
for tg in trade_glob_patterns:
    found = glob.glob(tg)
    if found:
        trade_matches = found
        break
print(f"Found {len(book_matches)} book files, {len(trade_matches)} trade files")

if book_matches:
    bf = book_matches[0]
    try:
        bdf = pd.read_csv(bf, nrows=5)
        print(f"Book file {bf} columns:", list(bdf.columns))
        print(bdf.head(5).to_string(index=False))
    except Exception as e:
        print(f"Failed to read book file {bf}: {e}")

if trade_matches:
    tfm = trade_matches[0]
    try:
        tdf = pd.read_csv(tfm, nrows=5)
        print(f"Market trade file {tfm} columns:", list(tdf.columns))
        print(tdf.head(5).to_string(index=False))
    except Exception as e:
        print(f"Failed to read market trade file {tfm}: {e}")

print("DONE")
