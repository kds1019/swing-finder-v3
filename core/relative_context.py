"""
Relative context for each candidate: where it sits in its 52-week range, and whether its
sector is leading or lagging the market. INFORMATIONAL ONLY (added 2026-10-05) — raw numbers
for the Decision Agent's judgment plus two visibility flags (Near52wHigh, SectorLagging), not
a screener gate and not a ranking criterion. Neither input has been backtested in this repo;
if they ever move into the ranking, backtest them first (docs/strategy.md pattern).

Both come from Alpaca daily bars the pipeline already fetches (the 300-session per-ticker
lookback covers a full 252-session year) plus one extra batch for the 11 SPDR sector ETFs.
Deliberately NOT from Webull's new screener endpoints: Webull's sector taxonomy doesn't map
onto FMP's sector names (core.universe), its 52-week list is a ranked screen rather than a
per-ticker lookup, and it would put more of the run behind Webull's 2FA token.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

YEAR_SESSIONS = 252
# Fewer sessions than this and the "52-week" range is too short to mean anything.
MIN_SESSIONS_FOR_RANGE = 200

# FMP's sector names (core.universe "Sector") -> the SPDR Select Sector ETF that tracks it.
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

SECTOR_RS_WINDOW = 20        # sessions for the leading/lagging read
SECTOR_RS_SHORT_WINDOW = 5   # sessions for "is that improving or fading right now"
# Sector ETF 20-session return minus SPY's, in percentage points.
SECTOR_LEADING_PP, SECTOR_LAGGING_PP = 2.0, -2.0
# Near52wHigh: price within this % of the 52-week high (overhead supply close by).
NEAR_52W_HIGH_PCT = 5.0


def _ret_pct(close: pd.Series, n: int) -> Optional[float]:
    if close is None or len(close) <= n:
        return None
    prev, last = float(close.iloc[-1 - n]), float(close.iloc[-1])
    return None if prev <= 0 else (last / prev - 1) * 100


def compute_52w_position(df: Optional[pd.DataFrame]) -> dict:
    """{"High52w", "Low52w", "PctFrom52wHigh" (<= 0), "PctAbove52wLow" (>= 0),
    "Range52wPosition" (0 = at the low, 1 = at the high)} from the last YEAR_SESSIONS bars,
    using intraday High/Low. All None if there's too little history."""
    empty = {"High52w": None, "Low52w": None, "PctFrom52wHigh": None,
             "PctAbove52wLow": None, "Range52wPosition": None}
    if df is None or len(df) < MIN_SESSIONS_FOR_RANGE or not {"High", "Low", "Close"}.issubset(df.columns):
        return empty
    window = df.tail(YEAR_SESSIONS)
    hi, lo, px = float(window["High"].max()), float(window["Low"].min()), float(window["Close"].iloc[-1])
    if hi <= 0 or lo <= 0 or hi <= lo:
        return empty
    return {
        "High52w": round(hi, 2),
        "Low52w": round(lo, 2),
        "PctFrom52wHigh": round((px / hi - 1) * 100, 2),
        "PctAbove52wLow": round((px / lo - 1) * 100, 2),
        "Range52wPosition": round((px - lo) / (hi - lo), 3),
    }


def compute_sector_strength(etf_bars: dict, spy_bars: Optional[pd.DataFrame]) -> dict:
    """{FMP sector: {"etf", "etf_20d_pct", "spy_20d_pct", "rs_20d_pp", "rs_5d_pp", "label"}}
    for every sector whose ETF bars are available. label: "leading" / "inline" / "lagging"
    by rs_20d_pp vs SECTOR_LEADING_PP / SECTOR_LAGGING_PP. {} if SPY is unavailable."""
    if spy_bars is None or spy_bars.empty:
        return {}
    spy_close = spy_bars["Close"]
    spy20, spy5 = _ret_pct(spy_close, SECTOR_RS_WINDOW), _ret_pct(spy_close, SECTOR_RS_SHORT_WINDOW)
    out: dict = {}
    for sector, etf in SECTOR_ETFS.items():
        bars = etf_bars.get(etf)
        if bars is None or bars.empty:
            continue
        e20, e5 = _ret_pct(bars["Close"], SECTOR_RS_WINDOW), _ret_pct(bars["Close"], SECTOR_RS_SHORT_WINDOW)
        if e20 is None or spy20 is None:
            continue
        rs20 = e20 - spy20
        rs5 = (e5 - spy5) if (e5 is not None and spy5 is not None) else None
        label = ("leading" if rs20 >= SECTOR_LEADING_PP
                 else "lagging" if rs20 <= SECTOR_LAGGING_PP else "inline")
        out[sector] = {
            "etf": etf,
            "etf_20d_pct": round(e20, 2),
            "spy_20d_pct": round(spy20, 2),
            "rs_20d_pp": round(rs20, 2),
            "rs_5d_pp": round(rs5, 2) if rs5 is not None else None,
            "label": label,
        }
    return out


def attach_sector_strength(df: pd.DataFrame, sector_strength: dict) -> pd.DataFrame:
    """Adds SectorETF / SectorRS20dPP / SectorRS5dPP / SectorStrength columns by each row's
    Sector. Rows whose sector has no ETF (e.g. "Unknown") get None — never raises."""
    if df.empty or "Sector" not in df.columns:
        return df
    out = df.copy()
    rows = [sector_strength.get(s) or {} for s in out["Sector"]]
    out["SectorETF"] = [r.get("etf") for r in rows]
    out["SectorRS20dPP"] = [r.get("rs_20d_pp") for r in rows]
    out["SectorRS5dPP"] = [r.get("rs_5d_pp") for r in rows]
    out["SectorStrength"] = [r.get("label") for r in rows]
    return out
