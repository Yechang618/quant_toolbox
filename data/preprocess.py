"""
Order Book Data Preprocessing.

Provides functions for cleaning raw order book snapshots, converting them to
Parquet/CSV, and computing basic micro-structure features for model training.
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from util.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file of order book snapshots into a DataFrame.

    Each line must be a JSON object with keys ``timestamp``, ``bids``,
    ``asks``.  Bids/asks are lists of ``[price, quantity]`` pairs.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        A :class:`pandas.DataFrame` with one row per snapshot.
    """
    path = Path(path)
    logger.info("Loading JSONL from %s", path)
    df = pd.read_json(path, lines=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    logger.info("Loaded %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def drop_duplicates_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and rows with null timestamps.

    Args:
        df: Raw snapshot DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"]).dropna(subset=["timestamp", "bids", "asks"])
    logger.info("Cleaned: %d → %d rows", before, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _best_bid(bids: List[List[float]]) -> Optional[float]:
    """Return the best (highest) bid price."""
    return float(bids[0][0]) if bids else None


def _best_ask(asks: List[List[float]]) -> Optional[float]:
    """Return the best (lowest) ask price."""
    return float(asks[0][0]) if asks else None


def _mid_price(bids: List[List[float]], asks: List[List[float]]) -> Optional[float]:
    """Compute the mid-price from the best bid and ask."""
    bb = _best_bid(bids)
    ba = _best_ask(asks)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return None


def _spread(bids: List[List[float]], asks: List[List[float]]) -> Optional[float]:
    """Compute the bid-ask spread."""
    bb = _best_bid(bids)
    ba = _best_ask(asks)
    if bb is not None and ba is not None:
        return ba - bb
    return None


def _order_book_imbalance(
    bids: List[List[float]],
    asks: List[List[float]],
    levels: int = 5,
) -> Optional[float]:
    """Compute order-book imbalance over the top ``levels`` price levels.

    OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)

    Args:
        bids: List of ``[price, quantity]`` pairs (best bid first).
        asks: List of ``[price, quantity]`` pairs (best ask first).
        levels: Number of price levels to include.

    Returns:
        Imbalance ratio in ``[-1, 1]``, or *None* if data is insufficient.
    """
    bid_vol = sum(float(b[1]) for b in bids[:levels])
    ask_vol = sum(float(a[1]) for a in asks[:levels])
    total = bid_vol + ask_vol
    if total == 0:
        return None
    return (bid_vol - ask_vol) / total


def engineer_features(df: pd.DataFrame, levels: int = 5) -> pd.DataFrame:
    """Add micro-structure feature columns to the snapshot DataFrame.

    New columns added:
    - ``best_bid`` — best bid price
    - ``best_ask`` — best ask price
    - ``mid_price`` — mid-point price
    - ``spread`` — bid-ask spread
    - ``obi`` — order-book imbalance (top *levels* levels)

    Args:
        df: Cleaned snapshot DataFrame (must have ``bids`` and ``asks`` cols).
        levels: Number of order-book levels to use for imbalance.

    Returns:
        DataFrame with additional feature columns.
    """
    logger.info("Engineering features (levels=%d)", levels)
    df = df.copy()
    df["best_bid"] = df["bids"].apply(_best_bid)
    df["best_ask"] = df["asks"].apply(_best_ask)
    df["mid_price"] = df.apply(lambda r: _mid_price(r["bids"], r["asks"]), axis=1)
    df["spread"] = df.apply(lambda r: _spread(r["bids"], r["asks"]), axis=1)
    df["obi"] = df.apply(
        lambda r: _order_book_imbalance(r["bids"], r["asks"], levels=levels),
        axis=1,
    )
    return df


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def save_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a Parquet file.

    The ``bids`` and ``asks`` columns are serialised as JSON strings because
    Parquet does not natively support nested lists without a schema.

    Args:
        df: DataFrame to save.
        path: Destination ``.parquet`` file path.
    """
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ("bids", "asks"):
        if col in out.columns:
            out[col] = out[col].apply(_json.dumps)
    out.to_parquet(path, index=False)
    logger.info("Saved %d rows to Parquet: %s", len(df), path)


def save_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to a CSV file.

    Args:
        df: DataFrame to save.
        path: Destination ``.csv`` file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to CSV: %s", len(df), path)


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    input_path: Path,
    output_dir: Path,
    fmt: str = "parquet",
    levels: int = 5,
) -> pd.DataFrame:
    """End-to-end preprocessing pipeline.

    Loads raw JSONL data, cleans it, engineers features, and persists the
    result in the requested format.

    Args:
        input_path: Path to the raw JSONL file.
        output_dir: Directory for processed output files.
        fmt: Output format: ``"parquet"`` or ``"csv"``.
        levels: Order-book levels for imbalance computation.

    Returns:
        Processed :class:`pandas.DataFrame`.
    """
    df = load_jsonl(input_path)
    df = drop_duplicates_and_nulls(df)
    df = engineer_features(df, levels=levels)

    stem = Path(input_path).stem
    output_dir = Path(output_dir)

    if fmt == "parquet":
        save_to_parquet(df, output_dir / f"{stem}_processed.parquet")
    elif fmt == "csv":
        save_to_csv(df, output_dir / f"{stem}_processed.csv")
    else:
        raise ValueError(f"Unsupported format: {fmt!r}.  Choose 'parquet' or 'csv'.")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Order book data preprocessor")
    parser.add_argument("input", help="Path to raw JSONL file")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--fmt", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--levels", type=int, default=5)
    args = parser.parse_args()

    result = run_preprocessing(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        fmt=args.fmt,
        levels=args.levels,
    )
    print(result.head())
