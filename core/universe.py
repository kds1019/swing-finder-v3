"""
Universe loader and Alpaca batching helper.

The universe CSV is a manual export from the "SwingFinder Master Universe"
Google Sheet (945 tickers) — this module only reads/validates it, it does not
fetch from Google directly.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "Ticker",
    "Company Name",
    "Exchange",
    "Sector",
    "Industry",
    "Price",
    "Market Cap ($M)",
    "Volume",
]


def load_universe(csv_path: str) -> pd.DataFrame:
    """Load and validate the universe CSV. Raises a clear error if the file is
    missing or malformed rather than failing deep inside the pipeline."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Universe CSV not found at '{csv_path}'. Export the 'SwingFinder Master "
            f"Universe' Google Sheet to CSV and save it there (see README.md)."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Universe CSV at '{csv_path}' is missing required columns: {missing}")

    return df


def batch_tickers(tickers: list[str], batch_size: int = 85) -> list[list[str]]:
    """Split tickers into batches for Alpaca's multi-symbol bars endpoint.
    85/batch was confirmed working (87 symbols x 60 days in one call, no error) —
    the real constraint is Alpaca's 1MB response cap, not a point-count limit."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
