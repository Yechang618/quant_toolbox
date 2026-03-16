"""
General-purpose helper utilities.

Covers time processing and file I/O operations used across the project.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware :class:`datetime`.

    Returns:
        Current UTC datetime.
    """
    return datetime.now(tz=timezone.utc)


def ts_to_datetime(timestamp_ms: int) -> datetime:
    """Convert a Unix timestamp in milliseconds to a UTC datetime.

    Args:
        timestamp_ms: Unix timestamp in milliseconds (as returned by Binance).

    Returns:
        Timezone-aware UTC :class:`datetime`.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)


def datetime_to_ts(dt: datetime) -> int:
    """Convert a datetime to a Unix timestamp in milliseconds.

    Args:
        dt: A :class:`datetime` object (naive datetimes are treated as UTC).

    Returns:
        Unix timestamp in milliseconds.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        String such as ``"1h 23m 45s"``.
    """
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


class Timer:
    """Simple context-manager / manual timer.

    Example::

        with Timer() as t:
            do_something()
        print(f"Elapsed: {t.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* (and any missing parents) if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`Path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Union[str, Path]) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Serialise *obj* to a JSON file.

    Args:
        obj: JSON-serialisable Python object.
        path: Destination file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)


def iter_jsonl(path: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    """Iterate over records in a JSONL file one line at a time.

    Args:
        path: Path to the ``.jsonl`` file.

    Yields:
        Parsed dict for each non-empty line.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Args:
        path: Path to the ``.parquet`` file.

    Returns:
        :class:`pandas.DataFrame`.
    """
    return pd.read_parquet(path)


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """List files in *directory* matching *pattern*.

    Args:
        directory: Directory to search.
        pattern: Glob pattern (e.g. ``"*.jsonl"``).
        recursive: If *True*, search sub-directories as well.

    Returns:
        Sorted list of matching :class:`Path` objects.
    """
    d = Path(directory)
    glob_fn = d.rglob if recursive else d.glob
    return sorted(p for p in glob_fn(pattern) if p.is_file())


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve an environment variable with an optional default.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is not set.

    Returns:
        The value of the environment variable, or *default*.
    """
    return os.environ.get(key, default)
