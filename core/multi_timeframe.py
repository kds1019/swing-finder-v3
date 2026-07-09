"""
Multi-timeframe (daily/weekly) trend alignment — ported from swing-finder-v2's
utils/multi_timeframe.py.

Two deviations from the reference:
  1. The 4-hour timeframe branch is dropped entirely rather than ported. It was
     already permanently disabled in the reference (hardcoded early `return None`,
     with the real implementation below it as unreachable dead code, requiring a
     Tiingo IEX Real-Time add-on the app doesn't have) — porting a disabled branch
     verbatim would just carry the dead code forward.
  2. RSI here uses the single Wilder's-smoothing implementation from
     core.indicators.rsi, not the reference's second, inconsistent simple-rolling-
     mean RSI — see core/indicators.py's module docstring.

Since Alpaca is the market data source now (not a fetch-per-timeframe API), "daily"
data is just whatever bars MarketDataAgent already pulled; "weekly" is a resample
of those same bars, not a separate fetch.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict, Optional

from core.indicators import rsi as wilders_rsi


def resample_to_weekly(daily_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if daily_df is None or daily_df.empty:
        return None

    return (
        daily_df.set_index("Date")
        .resample("W")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .reset_index()
    )


def calculate_mtf_indicators(df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """EMA20/50, Wilder's RSI14, MACD, trend (Uptrend/Downtrend), momentum
    (Strong/Weak/Neutral) for the most recent bar. Needs >= 26 bars (MACD)."""
    if df is None or df.empty or len(df) < 26:
        return None

    close = df["Close"].astype(float)

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]

    rsi14 = wilders_rsi(close, 14).iloc[-1]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    trend = "Uptrend" if ema20 > ema50 else "Downtrend"
    if rsi14 > 60:
        momentum = "Strong"
    elif rsi14 < 40:
        momentum = "Weak"
    else:
        momentum = "Neutral"

    return {
        "ema20": round(float(ema20), 2),
        "ema50": round(float(ema50), 2),
        "rsi14": round(float(rsi14), 2),
        "macd": round(float(macd.iloc[-1]), 4),
        "macd_signal": round(float(macd_signal.iloc[-1]), 4),
        "macd_hist": round(float(macd_hist.iloc[-1]), 4),
        "trend": trend,
        "momentum": momentum,
        "price": round(float(close.iloc[-1]), 2),
    }


def get_multi_timeframe_analysis(daily_df: pd.DataFrame) -> Dict:
    """
    Daily/weekly trend alignment for a single ticker's already-fetched daily bars.

    alignment_score is the fraction of timeframes (daily, weekly) trending
    "Uptrend" — with only 2 timeframes it can only be 0, 50, or 100.
    """
    weekly_df = resample_to_weekly(daily_df)

    daily_indicators = calculate_mtf_indicators(daily_df) if daily_df is not None and not daily_df.empty else None
    weekly_indicators = calculate_mtf_indicators(weekly_df) if weekly_df is not None and not weekly_df.empty else None

    uptrend_count = 0
    total_timeframes = 0
    for indicators in [daily_indicators, weekly_indicators]:
        if indicators:
            total_timeframes += 1
            if indicators["trend"] == "Uptrend":
                uptrend_count += 1

    alignment_score = (uptrend_count / total_timeframes) * 100 if total_timeframes > 0 else 0

    recommendation = _generate_mtf_recommendation(daily_indicators, weekly_indicators, alignment_score)

    return {
        "daily": daily_indicators,
        "weekly": weekly_indicators,
        "alignment_score": round(alignment_score, 1),
        "recommendation": recommendation,
    }


def _generate_mtf_recommendation(daily: Optional[Dict], weekly: Optional[Dict], alignment: float) -> str:
    if not daily:
        return "Insufficient data for analysis"

    if alignment >= 100:
        return "STRONG BUY SIGNAL - All timeframes aligned bullish. High-conviction setup."
    elif alignment >= 66:
        if weekly and weekly["trend"] == "Uptrend":
            return "BUY SIGNAL - Weekly uptrend confirmed. Good swing trade setup."
        return "CAUTIOUS BUY - Mixed signals. Wait for weekly confirmation."
    elif alignment >= 33:
        if daily["trend"] == "Uptrend" and daily["momentum"] == "Strong":
            return "NEUTRAL - Daily strong but higher timeframes mixed. Short-term trade only."
        return "NEUTRAL - Mixed timeframe signals. Wait for clarity."
    else:
        if weekly and weekly["trend"] == "Downtrend":
            return "AVOID - Weekly downtrend. Not a good swing trade setup."
        return "WEAK SETUP - Most timeframes bearish. Look for better opportunities."
