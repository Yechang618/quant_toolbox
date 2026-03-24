#!/usr/bin/env python3
"""
script/run_factor_gen.py
Command-line interface for running the feature generation pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.preprocess import run_pipeline
from util.logger import get_logger
from util.config import settings

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate slippage research features from trade records'
    )
    
    parser.add_argument(
        '--trade-file', '-t',
        type=str,
        default='combined_*.csv',
        help='Pattern for trade record files (default: combined_*.csv)'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        action='append',
        help='Symbol to process (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--date', '-d',
        type=str,
        action='append',
        help='Date to process in YYYYMMDD format (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--market-root',
        type=Path,
        default=settings.market_processed_root,
        help='Root path for market processed data'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=settings.output_root,
        help='Root path for output preprocessed data'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without saving output files'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Update settings if needed
    if args.market_root:
        settings.market_processed_root = Path(args.market_root)
    if args.output_dir:
        settings.output_root = Path(args.output_dir)
        settings.output_mode0 = settings.output_root / "mode0"
        settings.output_mode2 = settings.output_root / "mode2"
    
    logger.info(f"Starting with args: {args}")
    
    try:
        run_pipeline(
            date_start='2026-01',
            date_end='2026-01',
            trade_file_pattern=args.trade_file,
            symbols=args.symbol,
            dates=args.date,
            dry_run=args.dry_run
        )
        logger.info("Pipeline completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())