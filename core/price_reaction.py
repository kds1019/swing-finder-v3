"""
Price context for the Decision Agent — computed in Python so the model reads the tape
instead of guessing it from headlines.

Motivated by a live miss (2026-09-30 scan, SRRK): the Decision Agent called Scholar Rock's
FDA approval a "recent" catalyst that "drove an ~12% single-day pop". The approval hit
9/11 intraday; the 12% was a Benzinga PREMARKET headline on 9/14. The regular session that
day closed -6.4%, and by the scan the stock was ~11% below its pre-approval close. The agent
had headlines but no daily bars, so it had no way to see the sell-the-news reaction — and
the approval was 19 days old, well past the prompt's own ~7-day "recent" definition.

Two things are attached per candidate:
  recent_daily_bars()          — the last N sessions: date, close, % change, relative volume
  annotate_news_with_reaction() — per news item: age_days, the session it hit, that
                                  session's close-to-close % and relative volume, and the %
                                  move from the pre-news close to the latest close
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

RECENT_BARS_SESSIONS = 30
REL_VOLUME_LOOKBACK = 20
_MARKET_TZ = "America/New_York"
_MARKET_CLOSE_HOUR = 16  # items published at/after 16:00 ET land on the next session


def recent_daily_bars(bars: Optional[pd.DataFrame], n: int = RECENT_BARS_SESSIONS) -> list[dict]:
    """[{date, close, chg_pct, rel_vol}, ...] oldest-first for the last `n` sessions.
    rel_vol = volume / prior-20-session average. [] if bars are unavailable. Note: a scan
    run during market hours includes today's still-forming bar as the last row."""
    if bars is None or bars.empty or not {"Date", "Close", "Volume"}.issubset(bars.columns):
        return []
    b = bars[["Date", "Close", "Volume"]].copy()
    b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
    b = b.sort_values("Date").reset_index(drop=True)
    b["chg_pct"] = b["Close"].pct_change() * 100
    avg_vol = b["Volume"].rolling(REL_VOLUME_LOOKBACK).mean().shift(1)
    b["rel_vol"] = b["Volume"] / avg_vol
    out = []
    for r in b.tail(n).itertuples():
        out.append({
            "date": str(r.Date.date()),
            "close": round(float(r.Close), 2),
            "chg_pct": None if pd.isna(r.chg_pct) else round(float(r.chg_pct), 2),
            "rel_vol": None if pd.isna(r.rel_vol) else round(float(r.rel_vol), 2),
        })
    return out


def _parse_ts(value) -> pd.Timestamp:
    """UTC timestamp; bare numbers are epoch MILLISECONDS (pandas' to_json default), not
    the nanoseconds pd.to_datetime would assume. NaT if unparseable."""
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _reaction_session_index(published_utc: pd.Timestamp, dates: pd.Series) -> Optional[int]:
    """Index into `dates` (sorted, normalized, naive) of the first regular session an item
    published at `published_utc` could move: the same day if it printed before the 16:00 ET
    close on a trading day, otherwise the next trading day. None if that session isn't in
    the bars yet (e.g. an item from after the latest bar)."""
    et = published_utc.tz_convert(_MARKET_TZ)
    day = pd.Timestamp(et.date())
    if et.hour >= _MARKET_CLOSE_HOUR:
        day = day + pd.Timedelta(days=1)
    idx = int(dates.searchsorted(day, side="left"))
    return idx if idx < len(dates) else None


def annotate_news_with_reaction(
    news_items: list[dict],
    bars: Optional[pd.DataFrame],
    now: Optional[pd.Timestamp] = None,
) -> list[dict]:
    """Returns a copy of `news_items` with, per item:
      age_days            — calendar days since publication
      reaction_session    — YYYY-MM-DD of the session the item could first move
      session_chg_pct     — that session's close vs the prior close (the ACTUAL reaction —
                            not whatever the headline claims, e.g. a premarket move)
      session_rel_vol     — that session's volume / prior-20-session average
      chg_since_pct       — latest close vs the close BEFORE the reaction session: has the
                            stock held, extended, or given back the move since
    Fields are None where bars don't cover the item. Items keep their original keys.
    Publication time comes from "Date" (Alpaca: naive-UTC ISO string, or epoch ms) or
    "date"/"publishedDate"."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    b = None
    if bars is not None and not bars.empty and {"Date", "Close", "Volume"}.issubset(bars.columns):
        b = bars[["Date", "Close", "Volume"]].copy()
        b["Date"] = pd.to_datetime(b["Date"]).dt.normalize()
        b = b.sort_values("Date").reset_index(drop=True)
        b["avg_vol"] = b["Volume"].rolling(REL_VOLUME_LOOKBACK).mean().shift(1)

    out = []
    for item in news_items:
        item = dict(item)
        raw = next((item.get(k) for k in ("Date", "date", "publishedDate") if item.get(k)), None)
        ts = _parse_ts(raw)
        item.update({"age_days": None, "reaction_session": None, "session_chg_pct": None,
                     "session_rel_vol": None, "chg_since_pct": None})
        if pd.isna(ts):
            out.append(item)
            continue
        item["age_days"] = int((now - ts).days)
        if b is not None:
            i = _reaction_session_index(ts, b["Date"])
            if i is not None and i > 0:
                prev_close = float(b["Close"].iloc[i - 1])
                close = float(b["Close"].iloc[i])
                last = float(b["Close"].iloc[-1])
                av = b["avg_vol"].iloc[i]
                item["reaction_session"] = str(b["Date"].iloc[i].date())
                item["session_chg_pct"] = round((close / prev_close - 1) * 100, 2)
                item["session_rel_vol"] = None if pd.isna(av) or av <= 0 else round(float(b["Volume"].iloc[i] / av), 2)
                item["chg_since_pct"] = round((last / prev_close - 1) * 100, 2)
        out.append(item)
    return out
