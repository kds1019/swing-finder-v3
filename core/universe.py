"""
Universe builder (live FMP screener) and Alpaca batching helper.

Universe membership is built live from FMP's /stable/company-screener endpoint
on every pipeline run, using price_min/price_max (server-side) plus min_volume,
min_dollar_volume and market_cap_min_musd (client-side) from config.settings as
the actual filter — these used to just describe a one-time manual CSV export
("SwingFinder Master Universe" Google Sheet) that was trusted as-is and could
silently drift out of sync with the settings meant to describe it. See
docs/strategy.md for what each floor is for.
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
    # FMP's "volume" is TODAY'S cumulative intraday volume — filtering the universe on it
    # made universe size depend on the time of day the scan ran (pre-market: prior-day
    # value, ~1100 names; 30 min into the session: volume-so-far, ~150 names). "avgVolume"
    # is the stable average daily volume — that's what a liquidity floor needs.
    "avgVolume": "Volume",
}


def _screen_exchange(session: requests.Session, api_key: str, exchange: str, settings) -> list[dict]:
    """Filters by price range only — NOT volumeMoreThan too. Two reasons: (1) server-side
    volumeMoreThan filters on FMP's `volume` field, which is TODAY'S intraday cumulative
    volume, so it would make universe membership depend on the time of day the scan runs —
    build_universe() filters client-side on `avgVolume` (stable average daily volume)
    instead; (2) confirmed live 2026-08-24 that combining a price range AND volumeMoreThan
    in one request silently collapsed NYSE from 561 real matches to 20 rows (an FMP-side
    regression; the combined filter worked as recently as 2026-08-23)."""
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
    settings.price_min/price_max/min_volume/min_dollar_volume/market_cap_min_musd
    gate this directly (plus isActivelyTrading=true, isEtf=false, isFund=false,
    country=US) — they are the real filter now, not just descriptive of a stale
    CSV. Raises rather than silently returning an empty/partial universe.

    price_min/price_max are applied server-side (see _screen_exchange); min_volume,
    min_dollar_volume and market_cap_min_musd are applied here, client-side, after the
    fact. Volume moved client-side because combining it with the price filter in the same
    FMP request is currently broken there (see _screen_exchange's docstring); dollar
    volume and market cap are derived quantities that FMP's screener can't express
    directly anyway. Downstream (technical screener, then research / catalyst detection in
    the Decision Agent) is unaffected either way."""
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

    # "Volume" here is FMP's avgVolume (see _FIELD_RENAME) — stable average daily volume,
    # not the intraday figure that made this filter time-of-day dependent.
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    pre_volume_count = len(df)
    df = df[df["Volume"] >= settings.min_volume]
    print(f"[universe] After avg-volume filter (>= {settings.min_volume}): "
          f"{len(df)} / {pre_volume_count} tickers", file=sys.stderr)

    # Dollar volume (Price * avg Volume) is the meaningful liquidity unit — the share-count
    # floor above is kept as a secondary check. See docs/strategy.md.
    pre_dollar_vol_count = len(df)
    df = df[df["Price"] * df["Volume"] >= settings.min_dollar_volume]
    print(f"[universe] After dollar-volume filter (>= ${settings.min_dollar_volume:,.0f}): "
          f"{len(df)} / {pre_dollar_vol_count} tickers", file=sys.stderr)

    # NaN market cap (FMP returned null/0) fails this comparison and is dropped — an
    # unknown-size company is exactly the kind this floor exists to exclude.
    pre_mktcap_count = len(df)
    df = df[df["Market Cap ($M)"] >= settings.market_cap_min_musd]
    print(f"[universe] After market-cap filter (>= ${settings.market_cap_min_musd:,.0f}M): "
          f"{len(df)} / {pre_mktcap_count} tickers", file=sys.stderr)

    if df.empty:
        raise RuntimeError(
            "Universe is empty after the client-side liquidity / market-cap filters — "
            "check min_volume / min_dollar_volume / market_cap_min_musd in config.settings."
        )

    return df[REQUIRED_COLUMNS].reset_index(drop=True)


def batch_tickers(tickers: list[str], batch_size: int = 85) -> list[list[str]]:
    """Split tickers into batches for Alpaca's multi-symbol bars endpoint.
    85/batch was confirmed working (87 symbols x 60 days in one call, no error) —
    the real constraint is Alpaca's 1MB response cap, not a point-count limit."""
    return [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
