"""
Binance Order Book Downloader.

Downloads order book snapshots from Binance via REST API (ccxt) and streams
real-time updates via WebSocket.  Includes retry logic and rate-limit handling.

WARNING: Never hard-code API keys.  Load them from environment variables or a
         .env file (see util/config.py).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import ccxt
import ccxt.pro as ccxtpro

from util.config import settings
from util.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# REST helpers
# ---------------------------------------------------------------------------


def _build_exchange(
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
) -> ccxt.binance:
    """Create and configure a ccxt Binance exchange instance.

    Args:
        api_key: Binance API key.  If *None*, uses ``settings.BINANCE_API_KEY``.
        api_secret: Binance API secret.  If *None*, uses
            ``settings.BINANCE_API_SECRET``.

    Returns:
        A configured :class:`ccxt.binance` exchange object.
    """
    return ccxt.binance(
        {
            "apiKey": api_key or settings.BINANCE_API_KEY,
            "secret": api_secret or settings.BINANCE_API_SECRET,
            "enableRateLimit": True,  # honour Binance rate limits automatically
        }
    )


def fetch_order_book_snapshot(
    symbol: str,
    depth: int = 20,
    max_retries: int = 5,
    backoff_base: float = 1.5,
) -> Dict[str, Any]:
    """Fetch a single order book snapshot via the REST API.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        depth: Number of price levels to fetch on each side.
        max_retries: Maximum number of retry attempts on transient errors.
        backoff_base: Base for exponential back-off (seconds).

    Returns:
        Dictionary with keys ``timestamp``, ``bids``, ``asks``.

    Raises:
        ccxt.NetworkError: After exhausting all retry attempts.
        ccxt.ExchangeError: On non-retryable exchange errors.
    """
    exchange = _build_exchange()
    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(1, max_retries + 1):
        try:
            raw = exchange.fetch_order_book(symbol, limit=depth)
            return {
                "timestamp": raw.get("timestamp") or int(time.time() * 1000),
                "bids": raw["bids"],
                "asks": raw["asks"],
            }
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            last_error = exc
            wait = backoff_base**attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s – retrying in %.1fs",
                attempt,
                max_retries,
                symbol,
                exc,
                wait,
            )
            time.sleep(wait)
        except ccxt.RateLimitExceeded as exc:
            last_error = exc
            wait = backoff_base**attempt * 2
            logger.warning("Rate limit exceeded – waiting %.1fs", wait)
            time.sleep(wait)
        except ccxt.ExchangeError as exc:
            logger.error("Non-retryable exchange error: %s", exc)
            raise

    raise last_error


def collect_snapshots(
    symbol: str,
    output_dir: Path,
    num_snapshots: int = 100,
    interval_seconds: float = 1.0,
    depth: int = 20,
) -> List[Dict[str, Any]]:
    """Collect multiple order book snapshots at a fixed interval.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        output_dir: Directory where a JSONL file will be written.
        num_snapshots: How many snapshots to capture.
        interval_seconds: Pause (in seconds) between consecutive fetches.
        depth: Number of price levels per side.

    Returns:
        List of snapshot dictionaries that were captured.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("/", "_")
    out_path = output_dir / f"{safe_symbol}_orderbook.jsonl"

    snapshots: List[Dict[str, Any]] = []
    logger.info(
        "Starting REST collection: %d snapshots of %s → %s",
        num_snapshots,
        symbol,
        out_path,
    )

    with out_path.open("a", encoding="utf-8") as fh:
        for i in range(num_snapshots):
            snapshot = fetch_order_book_snapshot(symbol, depth=depth)
            snapshots.append(snapshot)
            fh.write(json.dumps(snapshot) + "\n")

            logger.debug("Snapshot %d/%d captured", i + 1, num_snapshots)
            if i < num_snapshots - 1:
                time.sleep(interval_seconds)

    logger.info("Collection complete – saved %d snapshots to %s", len(snapshots), out_path)
    return snapshots


# ---------------------------------------------------------------------------
# WebSocket streaming (ccxt.pro)
# ---------------------------------------------------------------------------


async def stream_order_book(
    symbol: str,
    output_dir: Path,
    duration_seconds: float = 60.0,
    depth: int = 20,
) -> None:
    """Stream order book updates via WebSocket and persist to JSONL.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        output_dir: Directory where the streaming output file will be written.
        duration_seconds: Total seconds to stream before closing.
        depth: Number of price levels per side.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_symbol = symbol.replace("/", "_")
    out_path = output_dir / f"{safe_symbol}_orderbook_ws.jsonl"

    exchange = ccxtpro.binance(
        {
            "apiKey": settings.BINANCE_API_KEY,
            "secret": settings.BINANCE_API_SECRET,
            "enableRateLimit": True,
        }
    )

    end_time = time.monotonic() + duration_seconds
    logger.info(
        "Starting WebSocket stream for %s – will run for %.0fs → %s",
        symbol,
        duration_seconds,
        out_path,
    )

    try:
        with out_path.open("a", encoding="utf-8") as fh:
            while time.monotonic() < end_time:
                raw = await exchange.watch_order_book(symbol, limit=depth)
                record: Dict[str, Any] = {
                    "timestamp": raw.get("timestamp") or int(time.time() * 1000),
                    "bids": raw["bids"],
                    "asks": raw["asks"],
                }
                fh.write(json.dumps(record) + "\n")
    finally:
        await exchange.close()
        logger.info("WebSocket stream closed.")


def run_websocket_stream(
    symbol: str,
    output_dir: Path,
    duration_seconds: float = 60.0,
    depth: int = 20,
) -> None:
    """Convenience wrapper to run the async WebSocket stream from sync code.

    Args:
        symbol: Trading pair, e.g. ``"BTC/USDT"``.
        output_dir: Directory where the streaming output file will be written.
        duration_seconds: Total seconds to stream before closing.
        depth: Number of price levels per side.
    """
    asyncio.run(
        stream_order_book(
            symbol=symbol,
            output_dir=output_dir,
            duration_seconds=duration_seconds,
            depth=depth,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binance order-book downloader")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument(
        "--mode",
        choices=["rest", "ws"],
        default="rest",
        help="Download mode: REST snapshots or WebSocket stream",
    )
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--num-snapshots", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60.0, help="WS duration (s)")
    parser.add_argument("--depth", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "rest":
        collect_snapshots(
            symbol=args.symbol,
            output_dir=Path(args.output_dir),
            num_snapshots=args.num_snapshots,
            depth=args.depth,
        )
    else:
        run_websocket_stream(
            symbol=args.symbol,
            output_dir=Path(args.output_dir),
            duration_seconds=args.duration,
            depth=args.depth,
        )
