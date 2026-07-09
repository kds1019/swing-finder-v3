"""
Relative strength vs SPY — ported from swing-finder-v2's utils/relative_strength.py,
stripped of @st.cache_data (no Streamlit here) and of internal Tiingo fetches: callers
pass in already-fetched ticker/SPY bar DataFrames (from MarketDataAgent) instead.
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any, Optional


def calculate_relative_strength_rank(
    ticker: str, ticker_df: Optional[pd.DataFrame], spy_df: Optional[pd.DataFrame], period: int = 60
) -> Optional[Dict[str, Any]]:
    """Relative strength vs SPY over `period` bars. rs_ratio is a percentage-point
    difference in returns (ticker_return - spy_return), not a ratio despite the name
    (kept as-is from the reference for continuity)."""
    if ticker_df is None or spy_df is None or len(ticker_df) < period or len(spy_df) < period:
        return None

    ticker_recent = ticker_df.tail(period)
    spy_recent = spy_df.tail(period)

    ticker_return = (ticker_recent["Close"].iloc[-1] / ticker_recent["Close"].iloc[0] - 1) * 100
    spy_return = (spy_recent["Close"].iloc[-1] / spy_recent["Close"].iloc[0] - 1) * 100

    rs_ratio = ticker_return - spy_return

    if len(ticker_recent) >= 60:
        recent_20 = ticker_recent.tail(20)
        previous_40 = ticker_recent.iloc[-60:-20]
        recent_return = (recent_20["Close"].iloc[-1] / recent_20["Close"].iloc[0] - 1) * 100
        previous_return = (previous_40["Close"].iloc[-1] / previous_40["Close"].iloc[0] - 1) * 100
        momentum = recent_return - previous_return
    else:
        momentum = 0

    if rs_ratio > 10:
        strength, emoji = "Very Strong", "\U0001F525"
    elif rs_ratio > 5:
        strength, emoji = "Strong", "\U0001F4AA"
    elif rs_ratio > 0:
        strength, emoji = "Above Market", "✅"
    elif rs_ratio > -5:
        strength, emoji = "Below Market", "⚠️"
    else:
        strength, emoji = "Weak", "❌"

    return {
        "ticker": ticker,
        "ticker_return": round(ticker_return, 2),
        "spy_return": round(spy_return, 2),
        "rs_ratio": round(rs_ratio, 2),
        "momentum": round(momentum, 2),
        "strength": strength,
        "emoji": emoji,
    }


def rank_watchlist_by_strength(
    ticker_bars: Dict[str, pd.DataFrame], spy_df: pd.DataFrame, period: int = 60
) -> List[Dict[str, Any]]:
    """ticker_bars: {ticker: bars_df} for all tickers to rank, e.g. the SmartScore shortlist."""
    results = []
    for ticker, df in ticker_bars.items():
        rs_data = calculate_relative_strength_rank(ticker, df, spy_df, period)
        if rs_data:
            results.append(rs_data)
    results.sort(key=lambda x: x["rs_ratio"], reverse=True)
    return results


def get_top_performers(ticker_bars: Dict[str, pd.DataFrame], spy_df: pd.DataFrame, period: int = 60, top_n: int = 10) -> List[Dict[str, Any]]:
    return rank_watchlist_by_strength(ticker_bars, spy_df, period)[:top_n]


def get_bottom_performers(ticker_bars: Dict[str, pd.DataFrame], spy_df: pd.DataFrame, period: int = 60, bottom_n: int = 10) -> List[Dict[str, Any]]:
    return rank_watchlist_by_strength(ticker_bars, spy_df, period)[-bottom_n:]


def calculate_rs_score(rs_ratio: float, momentum: float) -> int:
    """RS score (0-100): 0-60 points from rs_ratio, 0-40 from momentum."""
    if rs_ratio > 20:
        rs_points = 60
    elif rs_ratio > 10:
        rs_points = 50
    elif rs_ratio > 5:
        rs_points = 40
    elif rs_ratio > 0:
        rs_points = 30
    elif rs_ratio > -5:
        rs_points = 20
    else:
        rs_points = 10

    if momentum > 10:
        momentum_points = 40
    elif momentum > 5:
        momentum_points = 30
    elif momentum > 0:
        momentum_points = 20
    elif momentum > -5:
        momentum_points = 10
    else:
        momentum_points = 0

    return rs_points + momentum_points


def get_multi_timeframe_strength(
    ticker: str, ticker_df: Optional[pd.DataFrame], spy_df: Optional[pd.DataFrame]
) -> Optional[Dict[str, Any]]:
    """RS ratio across a fixed set of lookback windows (1wk/1mo/3mo/6mo/1yr).
    Requires ticker_df/spy_df to have at least 250 bars of history for the 1-year window."""
    if ticker_df is None or spy_df is None:
        return None

    timeframes = {"1_week": 5, "1_month": 20, "3_months": 60, "6_months": 120, "1_year": 250}
    results = {}
    for name, period in timeframes.items():
        rs_data = calculate_relative_strength_rank(ticker, ticker_df, spy_df, period)
        if rs_data:
            results[name] = {"rs_ratio": rs_data["rs_ratio"], "strength": rs_data["strength"], "emoji": rs_data["emoji"]}
    return results


def analyze_strength_trend(multi_tf_data: Optional[Dict[str, Any]]) -> str:
    """Is relative strength accelerating, improving, stable, weakening, or decelerating?"""
    if not multi_tf_data:
        return "Unknown"

    timeframes = ["1_week", "1_month", "3_months", "6_months", "1_year"]
    rs_values = [multi_tf_data[tf]["rs_ratio"] for tf in timeframes if tf in multi_tf_data]

    if len(rs_values) < 3:
        return "Insufficient data"

    short_term_avg = sum(rs_values[:2]) / 2 if len(rs_values) >= 2 else rs_values[0]
    long_term_avg = sum(rs_values[-2:]) / 2 if len(rs_values) >= 2 else rs_values[-1]

    if short_term_avg > long_term_avg + 5:
        return "Accelerating (Getting Stronger)"
    elif short_term_avg > long_term_avg:
        return "Improving"
    elif short_term_avg < long_term_avg - 5:
        return "Decelerating (Getting Weaker)"
    elif short_term_avg < long_term_avg:
        return "Weakening"
    else:
        return "Stable"
