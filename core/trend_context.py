"""
Trend-state + swing-structure Fib retracement — an ADDITIONAL context layer on top of
core.pullback_reversal's short-term stabilization signal (higher low / contracting range /
falling down-up volume ratio). That signal says a pullback has stopped falling; it says
nothing about whether it's stopping *inside an uptrend* (a continuation, room to run) or
*inside a downtrend* (a bounce that may just be relief before the next leg down). The old
single "confirmed support" read conflated the two.

Motivating case (2026-09-11): the live scan flagged CRUS on exactly the short-term signal
(higher low, contracting range, falling down/up volume) while it was still below both its
50-day and 200-day EMA, only ~30% retraced off its down-leg — a reversion bounce, not a
confirmed pullback in an uptrend. That's possible because core.pullback_reversal's own
uptrend gate uses a much longer slope window (EMA200 risen >=5% over the last *126*
sessions) than this module's 20-session slope — a ticker can clear that slower gate while,
by this module's more current read, still be rolling over into a downtrend. That's
intentional here: this module adds the more current read as a SEPARATE, informational axis,
not a replacement for the screener's gate.

Uses EMA, not SMA (switched 2026-09-11 after a live case, RDW: SMA200 still read a mild
uptrend while price had already fallen back below both EMA50/EMA200 and EMA200's own
20-session slope had gone flat — EMA reacts faster to a recent stall/rollover because it
weights recent bars more heavily, which is exactly the property this read wants at the
inflection points where it matters most; SMA's slower reaction was masking exactly the kind
of rollover this module exists to catch). Reuses core.indicators.compute_indicators()'s
EMA50/EMA200 columns rather than recomputing them — same convention as
core.pullback_reversal's own functions, which likewise require compute_indicators() to have
already run.

Deliberately NOT wired into the screener gate, live ranking, or position sizing yet — this
is Phase 1 (compute + log every field so setup_type below can be backtested per-bucket,
see research/trend_context_backtest.py) of the two-phase plan in docs/strategy.md's trend
context section. Phase 2 backtests trend_continuation vs reversion_bounce separately;
wiring either into live picks/sizing is deferred until that backtest justifies it.
"""

from __future__ import annotations

import pandas as pd

from core.indicators import find_pivot_points

# How far back the 200-EMA slope is measured to call it "rising" / "falling". Deliberately
# much shorter than core.pullback_reversal's 126-day EMA200 trend check — the whole point of
# this read is to catch a more current rollover/reclaim than that slower gate does.
EMA200_SLOPE_LOOKBACK_DAYS = 20
# Slope smaller than this (as %, either sign) counts as flat rather than clearly
# rising/falling, and falls through to "transitional" instead of being forced into
# uptrend/downtrend on noise.
EMA200_SLOPE_FLAT_BAND_PCT = 0.5

# Long-horizon (1-year) EMA200 slope — informational only, NOT a gate (an isolated backtest,
# research/long_horizon_gate_ab.py, tested rejecting on this and it made every window worse:
# it can't tell a recovery that's stalling apart from one that's genuinely continuing, so
# blocking both loses more than it saves). Exists so the Decision Agent can SEE the
# discrepancy and judge case-by-case: a live case (ENPH, 2026-09-11) had a strong 126-day
# slope (+7.5%, "uptrend" per the screener gate) built almost entirely from a sharp recovery
# off a low ~5 months back, while the 252-day slope was only +2.4% — a V-shaped round trip
# stalling at its own recent high, not a sustained trend, invisible to any single two-point
# slope comparison alone. Requires more history than compute_trend_state()'s own 220-bar
# minimum, so it's computed leniently (just needs len(df) > this lookback) and degrades to
# None on its own rather than failing the whole function when unavailable.
EMA200_LONG_SLOPE_LOOKBACK_DAYS = 252

MIN_BARS_FOR_TREND_STATE = 200 + EMA200_SLOPE_LOOKBACK_DAYS


def compute_trend_state(df: pd.DataFrame) -> dict:
    """EMA50/EMA200 read for the most recent bar of `df` — requires compute_indicators() to
    have already been applied (needs the EMA50/EMA200 columns it adds), same requirement as
    core.pullback_reversal's functions. Returns {} if there isn't enough history (needs
    200 + EMA200_SLOPE_LOOKBACK_DAYS bars) or those columns aren't present. Fields:
      ema50 / ema200
      price_above_ema50 / price_above_ema200 — bool
      ema200_slope_pct — % change in EMA200 over the last EMA200_SLOPE_LOOKBACK_DAYS sessions
      ema200_long_slope_pct — % change in EMA200 over the last EMA200_LONG_SLOPE_LOOKBACK_DAYS
          (252) sessions — None if there isn't enough history. Informational only; see the
          constant's comment above for why this isn't a gate. A much weaker long_slope than
          ema200_uptrend_pct (core.pullback_reversal's 126-day version) is the signature of a
          recovery stalling at its own recent high rather than a genuine sustained trend.
      trend_state — one of:
        "uptrend"      price above EMA200 AND EMA200 clearly rising
        "downtrend"    price below EMA200 AND EMA200 clearly falling
        "transitional" anything else — e.g. bounced off lows but hasn't reclaimed the
                       50-day yet, or EMA200 is flat, or price/EMA200 direction disagree
    """
    if (
        df is None or len(df) < MIN_BARS_FOR_TREND_STATE
        or "EMA50" not in df.columns or "EMA200" not in df.columns
    ):
        return {}

    close = df["Close"].astype(float)
    ema50 = df["EMA50"].astype(float)
    ema200 = df["EMA200"].astype(float)

    px = float(close.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    ema200_now = float(ema200.iloc[-1])
    ema200_then = float(ema200.iloc[-1 - EMA200_SLOPE_LOOKBACK_DAYS])
    if pd.isna(ema50_now) or pd.isna(ema200_now) or pd.isna(ema200_then) or ema200_then <= 0:
        return {}

    ema200_slope_pct = round((ema200_now - ema200_then) / ema200_then * 100, 2)
    price_above_ema50 = px > ema50_now
    price_above_ema200 = px > ema200_now

    if price_above_ema200 and ema200_slope_pct > EMA200_SLOPE_FLAT_BAND_PCT:
        trend_state = "uptrend"
    elif (not price_above_ema200) and ema200_slope_pct < -EMA200_SLOPE_FLAT_BAND_PCT:
        trend_state = "downtrend"
    else:
        trend_state = "transitional"

    ema200_long_slope_pct = None
    if len(ema200) > EMA200_LONG_SLOPE_LOOKBACK_DAYS:
        ema200_long_ago = float(ema200.iloc[-1 - EMA200_LONG_SLOPE_LOOKBACK_DAYS])
        if not pd.isna(ema200_long_ago) and ema200_long_ago > 0:
            ema200_long_slope_pct = round(
                (ema200_now - ema200_long_ago) / ema200_long_ago * 100, 2
            )

    return {
        "ema50": round(ema50_now, 2),
        "ema200": round(ema200_now, 2),
        "price_above_ema50": price_above_ema50,
        "price_above_ema200": price_above_ema200,
        "ema200_slope_pct": ema200_slope_pct,
        "ema200_long_slope_pct": ema200_long_slope_pct,
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
