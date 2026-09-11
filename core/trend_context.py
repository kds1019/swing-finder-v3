"""
Trend-state + swing-structure Fib retracement — an ADDITIONAL context layer on top of
core.pullback_reversal's short-term stabilization signal (higher low / contracting range /
falling down-up volume ratio). That signal says a pullback has stopped falling; it says
nothing about whether it's stopping *inside an uptrend* (a continuation, room to run) or
*inside a downtrend* (a bounce that may just be relief before the next leg down). The old
single "confirmed support" read conflated the two.

Motivating case (2026-09-11): the live scan flagged CRUS on exactly the short-term signal
(higher low, contracting range, falling down/up volume) while it was still below both its
50-day and 200-day SMA, only ~30% retraced off its down-leg — a reversion bounce, not a
confirmed pullback in an uptrend. That's possible because core.pullback_reversal's own
uptrend gate is a slower, different read (EMA200 risen >=5% over the last *126* sessions,
price within a wide band of that EMA) than a literal current SMA50/SMA200 position/slope
check — a ticker can clear the EMA200 gate while, by this module's plainer read, still be in
a downtrend. That's intentional here: this module adds the plainer read as a SEPARATE,
informational axis, not a replacement for the screener's gate.

Deliberately NOT wired into the screener gate, live ranking, or position sizing yet — this
is Phase 1 (compute + log every field so setup_type below can be backtested per-bucket,
see research/trend_context_backtest.py) of the two-phase plan in docs/strategy.md's trend
context section. Phase 2 backtests trend_continuation vs reversion_bounce separately;
wiring either into live picks/sizing is deferred until that backtest justifies it.
"""

from __future__ import annotations

import pandas as pd

from core.indicators import find_pivot_points

# How far back the 200-SMA slope is measured to call it "rising" / "falling". Deliberately
# much shorter than core.pullback_reversal's 126-day EMA200 trend check — the whole point of
# this read is to catch a more current rollover/reclaim than that slower gate does.
SMA200_SLOPE_LOOKBACK_DAYS = 20
# Slope smaller than this (as %, either sign) counts as flat rather than clearly
# rising/falling, and falls through to "transitional" instead of being forced into
# uptrend/downtrend on noise.
SMA200_SLOPE_FLAT_BAND_PCT = 0.5

MIN_BARS_FOR_TREND_STATE = 200 + SMA200_SLOPE_LOOKBACK_DAYS


def compute_trend_state(df: pd.DataFrame) -> dict:
    """SMA50/SMA200 read for the most recent bar of `df`. Returns {} if there isn't enough
    history (needs 200 + SMA200_SLOPE_LOOKBACK_DAYS bars). Fields:
      sma50 / sma200
      price_above_sma50 / price_above_sma200 — bool
      sma200_slope_pct — % change in SMA200 over the last SMA200_SLOPE_LOOKBACK_DAYS sessions
      trend_state — one of:
        "uptrend"      price above SMA200 AND SMA200 clearly rising
        "downtrend"    price below SMA200 AND SMA200 clearly falling
        "transitional" anything else — e.g. bounced off lows but hasn't reclaimed the
                       50-day yet, or SMA200 is flat, or price/SMA200 direction disagree
    """
    if df is None or len(df) < MIN_BARS_FOR_TREND_STATE:
        return {}

    close = df["Close"].astype(float)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    px = float(close.iloc[-1])
    sma50_now = float(sma50.iloc[-1])
    sma200_now = float(sma200.iloc[-1])
    sma200_then = float(sma200.iloc[-1 - SMA200_SLOPE_LOOKBACK_DAYS])
    if pd.isna(sma50_now) or pd.isna(sma200_now) or pd.isna(sma200_then) or sma200_then <= 0:
        return {}

    sma200_slope_pct = round((sma200_now - sma200_then) / sma200_then * 100, 2)
    price_above_sma50 = px > sma50_now
    price_above_sma200 = px > sma200_now

    if price_above_sma200 and sma200_slope_pct > SMA200_SLOPE_FLAT_BAND_PCT:
        trend_state = "uptrend"
    elif (not price_above_sma200) and sma200_slope_pct < -SMA200_SLOPE_FLAT_BAND_PCT:
        trend_state = "downtrend"
    else:
        trend_state = "transitional"

    return {
        "sma50": round(sma50_now, 2),
        "sma200": round(sma200_now, 2),
        "price_above_sma50": price_above_sma50,
        "price_above_sma200": price_above_sma200,
        "sma200_slope_pct": sma200_slope_pct,
        "trend_state": trend_state,
    }


# Swing-structure Fib retracement — separate from core.indicators.calculate_fibonacci_levels,
# which uses a 20-bar lookback tuned for the short-term pullback shape already used elsewhere
# in this screener. This looks for the actual multi-week leg instead: 60 sessions (~3 months)
# is long enough to capture a real swing without being so long it reacts to last quarter's
# structure instead of this one — matches the volume-profile window already used in
# core.pullback_reversal. Wider left/right bars than the short-term helper (5 vs 2) so the
# pivot search anchors on a genuinely significant high/low, not a single-bar spike.
SWING_LOOKBACK_DAYS = 60
SWING_PIVOT_LEFT_RIGHT_BARS = 5

# Classic retracement zone: price has given back 38.2%-61.8% of the most recent swing
# high-to-low leg. Used to flag a pullback-in-uptrend as sitting in the zone where a
# continuation entry is conventionally taken (see docs/strategy.md).
FIB_ZONE_MIN_PCT = 38.2
FIB_ZONE_MAX_PCT = 61.8


def measure_swing_fib_retracement(df: pd.DataFrame, lookback: int = SWING_LOOKBACK_DAYS) -> dict:
    """Most recent major swing high/low over `lookback` sessions (via
    core.indicators.find_pivot_points, wider left/right bars than its short-term caller to
    avoid anchoring on noise) and how far current price has retraced from that high back
    toward that low. Returns {} if there isn't enough history or the swing range is zero.
    Fields:
      swing_high / swing_low
      retracement_pct — % of the high-to-low range given back from the high, 0 at
          swing_high, 100 at swing_low (clamped to [0, 100]) — same convention as the
          "38.2%"/"61.8%" fib_levels already used in core.indicators.calculate_fibonacci_levels
      in_fib_zone — bool, retracement_pct within [FIB_ZONE_MIN_PCT, FIB_ZONE_MAX_PCT]
    """
    if df is None or len(df) < lookback:
        return {}

    window = df.tail(lookback).reset_index(drop=True)
    pivots = find_pivot_points(
        window, left_bars=SWING_PIVOT_LEFT_RIGHT_BARS, right_bars=SWING_PIVOT_LEFT_RIGHT_BARS,
    )
    phs, pls = pivots["pivot_highs"], pivots["pivot_lows"]

    swing_high = float(max(phs, key=lambda p: p["bar"])["price"]) if phs else float(window["High"].max())
    swing_low = float(max(pls, key=lambda p: p["bar"])["price"]) if pls else float(window["Low"].min())

    swing_range = swing_high - swing_low
    if swing_range <= 0:
        return {}

    px = float(df["Close"].iloc[-1])
    retracement_pct = round(max(0.0, min(100.0, (swing_high - px) / swing_range * 100)), 2)

    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "retracement_pct": retracement_pct,
        "in_fib_zone": FIB_ZONE_MIN_PCT <= retracement_pct <= FIB_ZONE_MAX_PCT,
    }


def classify_setup_type(
    trend_state: str | None, in_fib_zone: bool | None, stabilization_signal: bool
) -> str | None:
    """setup_type per the trend-continuation / reversion-bounce split (docs/strategy.md):
      "trend_continuation" — trend_state == "uptrend" AND in_fib_zone AND stabilization_signal
      "reversion_bounce"   — stabilization_signal True AND trend_state in
                              ("downtrend", "transitional")
      None — stabilization_signal is False (neither bucket applies — the short-term signal
             itself hasn't fired), or trend_state couldn't be computed (insufficient history)

    stabilization_signal is the caller's existing higher-low/contracting-range/volume read
    (core.pullback_reversal's KnifeRiskTier == "stabilising") — this module doesn't
    recompute it, so there's exactly one definition of "has it stabilised" shared between
    setup_type and support_status, not two that could quietly drift apart."""
    if trend_state is None or not stabilization_signal:
        return None
    if trend_state == "uptrend" and in_fib_zone:
        return "trend_continuation"
    if trend_state in ("downtrend", "transitional"):
        return "reversion_bounce"
    return None
