"""
Universe builder (live FMP screener) and Alpaca batching helper.

Universe membership is built live from FMP's /stable/company-screener endpoint
on every pipeline run, using price_min/price_max/min_volume from config.settings
as the actual filter — these used to just describe a one-time manual CSV export
("SwingFinder Master Universe" Google Sheet) that was trusted as-is and could
silently drift out of sync with the settings meant to describe it.
"""

from __future__ import annotations

import sys
from typing import Optional

import pandas as pd
import requests

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
EXCHANGES = ("NYSE", "NASDAQ", "AMEX")
PAGE_LIMIT = 1000
MAX_PAGES = 20  # safety net in case pagination doesn't behave as documented

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

_FIELD_RENAME = {
    "symbol": "Ticker",
    "companyName": "Company Name",
    "exchange": "Exchange",
    "sector": "Sector",
    "industry": "Industry",
    "price": "Price",
    "volume": "Volume",
}


def _screen_exchange(session: requests.Session, api_key: str, exchange: str, settings) -> list[dict]:
    rows: list[dict] = []
    for page in range(MAX_PAGES):
        params = {
            "apikey": api_key,
            "exchange": exchange,
            "country": "US",
            "isActivelyTrading": "true",
            "isEtf": "false",
            "isFund": "false",
            "priceMoreThan": settings.price_min,
            "priceLowerThan": settings.price_max,
            "volumeMoreThan": settings.min_volume,
            "limit": PAGE_LIMIT,
            "page": page,
        }
        resp = session.get(f"{FMP_BASE_URL}/company-screener", params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_LIMIT:
            break
    return rows


def build_universe(settings, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Builds the trading universe live from FMP's company-screener endpoint,
    querying NYSE/NASDAQ/AMEX separately and deduping by ticker symbol.
    settings.price_min/price_max/min_volume gate this directly (plus
    isActivelyTrading=true, isEtf=false, isFund=false, country=US) — they are
    the real filter now, not just descriptive of a stale CSV. Raises rather
    than silently returning an empty/partial universe."""
    if not settings.fmp_api_key:
        raise RuntimeError(
            "FMP_API_KEY is required to build the live universe. Add it to your .env."
        )

    sess = session or requests.Session()
    seen: dict[str, dict] = {}
    for exchange in EXCHANGES:
        rows = _screen_exchange(sess, settings.fmp_api_key, exchange, settings)
        print(f"[universe] {exchange}: {len(rows)} rows", file=sys.stderr)
        for row in rows:
            symbol = row.get("symbol")
            if symbol and symbol not in seen:
                seen[symbol] = row

    if not seen:
        raise RuntimeError(
            "FMP company-screener returned zero rows across NYSE/NASDAQ/AMEX — "
            "refusing to hand back an empty universe. Check FMP_API_KEY and the "
            "price_min/price_max/min_volume settings."
        )

    df = pd.DataFrame(seen.values())
    missing_source = [c for c in ["marketCap", *_FIELD_RENAME] if c not in df.columns]
    if missing_source:
        raise ValueError(f"FMP company-screener response is missing expected fields: {missing_source}")

    df = df.rename(columns=_FIELD_RENAME)
    df["Market Cap ($M)"] = df["marketCap"] / 1_000_000

    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def batch_tickers(tickers: list[str], batch_size: int = 85) -> list[list[str]]:
    """Split tickers into batches for Alpaca's multi-symbol bars endpoint.
    85/batch was confirmed working (87 symbols x 60 days in one call, no error) —
    the real constraint is Alpaca's 1MB response cap, not a point-count limit."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
