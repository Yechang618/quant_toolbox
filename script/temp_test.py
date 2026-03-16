# Temporary script, safe to delete.
"""
temp_test.py – Hello World / connection test.

This script verifies that the project environment is correctly set up by:
1. Printing a "Hello, World!" greeting.
2. Checking that required third-party packages are importable.
3. Optionally pinging the Binance public ticker endpoint (no auth required).

Temporary script, safe to delete.
"""

import sys


def hello_world() -> None:
    """Print a simple greeting to confirm the environment is alive."""
    print("Hello, World! quant_toolbox is ready.")


def check_imports() -> bool:
    """Verify that key dependencies are installed.

    Returns:
        *True* if all packages import successfully, *False* otherwise.
    """
    packages = ["ccxt", "pandas", "pydantic", "pydantic_settings", "pyarrow"]
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  [OK]  {pkg}")
        except ImportError:
            print(f"  [MISSING]  {pkg} – run: pip install {pkg}")
            all_ok = False
    return all_ok


def test_binance_connection() -> None:
    """Attempt a lightweight public REST call to Binance (no API key needed)."""
    try:
        import ccxt

        exchange = ccxt.binance({"enableRateLimit": True})
        ticker = exchange.fetch_ticker("BTC/USDT")
        print(f"  [OK]  Binance BTC/USDT last price: {ticker['last']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [WARN]  Could not reach Binance: {exc}")


if __name__ == "__main__":
    hello_world()
    print("\nChecking imports...")
    ok = check_imports()

    if "--skip-network" not in sys.argv:
        print("\nTesting Binance connection...")
        test_binance_connection()

    print("\nDone." if ok else "\nSome packages are missing – see above.")
