"""
EMA200 pullback/reversal screener — FRESH DESIGN, replaces core.smartscore's
classify_setup()-driven gating in the live scan path. Not a port of anything in
swing-finder-v2.

Built and calibrated directly against a real trade (EMBJ, bought 2026-06-12 at
$57.60, sold 2026-07-01 at $64.00, +11.3%), pulled from the user's actual Webull
order history and price bars rather than worked from memory of a description:
  - EMA200 rose from $42.74 (1yr before) -> $54.34 (6mo before) -> $61.36 (3mo
    before) -> $60.76 (at entry): a strong, sustained long-term uptrend that had
    only just flattened in the most recent 3 months — exactly what a real pullback
    inside an uptrend looks like, since a 200-day average reacts slowly and will
    read as "flat" during one even when the longer trend is genuinely up. This is
    why the uptrend check below looks 6 months back, not just the last few weeks.
  - Price at entry was $57.60 vs EMA200 $60.76: -5.2%, i.e. "slightly under."
  - The 15 trading days into the entry ranged roughly $53-59 (a ~10% band) with a
    local low around $53.63 two days before entry, then a two-day bounce into the
    $57.60 buy — "stabilized, then started a slight reversal up."
  - Volume profile (60-day window) at entry: POC $56.53, value area $54.28-$63.30.
    Price ($57.60) was actually just *above* the POC (+1.89%), not at/below it —
    an earlier draft of this screener required price <= POC and would have wrongly
    excluded the real reference trade. It was, however, comfortably inside the
    value area, well below value_area_high — that's the condition used below.

Every threshold here is a **percentage relative to that ticker's own price/EMA200/
volume profile**, not an absolute dollar level — this scans for the shape of the
setup (pullback into a rising 200-day average, stabilize, early bounce, not
stretched into thin volume above) at any price, not for stocks that resemble
EMBJ's price level specifically.

core.smartscore's classify_setup() (RSI/Bollinger-band-based Breakout/Pullback),
the ML-edge adjustment, and chart-pattern detection were walk-forward tested this
research effort and found no demonstrated edge (see
docs/ml-edge-confidence-research.md) — removed from the live pipeline entirely
rather than kept alongside this. Volume profile was never tested and is folded in
here per explicit instruction (untested, not disproven, unlike the other three).
This screener itself is not validated the same rigorous way per explicit
instruction either — treat its output as an unproven candidate signal.
"""

from __future__ import annotations

import pandas as pd

from core.volume_profile import compute_volume_profile

# How far back to check whether EMA200 itself is trending up — long enough that a
# genuine multi-week pullback (which flattens the recent EMA200 slope even inside a
# real uptrend, as EMBJ's own 3-months-before number shows) doesn't get misread as
# "no uptrend."
EMA200_TREND_LOOKBACK_DAYS = 126  # ~6 months of trading days
EMA200_MIN_UPTREND_PCT = 5.0  # EMBJ's own 6-month EMA200 gain was +11.8%; this leaves real margin

# Price must sit within this band of EMA200 — "slightly above or slightly under."
# EMBJ was -5.2%; banded asymmetrically since "pulled back" implies below more often,
# but the user's own wording allows slightly above too.
PRICE_VS_EMA200_MIN_PCT = -12.0
PRICE_VS_EMA200_MAX_PCT = 8.0

# Consolidation/stabilization window and thresholds — EMBJ's own 15-day range into
# entry was about 10% of price, with a bounce of ~7.4% off the window's low.
CONSOLIDATION_LOOKBACK_DAYS = 15
CONSOLIDATION_MAX_RANGE_PCT = 15.0
MIN_BOUNCE_OFF_LOW_PCT = 3.0

# Volume profile window — matches pipeline.py's prior VOLUME_PROFILE_WINDOW_DAYS.
# Gate is "not extended above the value area" (price <= value_area_high), not
# "at/below POC" — the latter would have excluded the real EMBJ trade itself.
VOLUME_PROFILE_WINDOW_DAYS = 60


def detect_pullback_reversal(df: pd.DataFrame) -> dict:
    """Detects the EMA200 pullback + stabilization/reversal setup for the most
    recent bar of `df` (must already have compute_indicators() applied — needs
    EMA200). Returns {"detected": False} if there isn't enough history or the
    setup's criteria aren't met; otherwise returns "detected": True plus the raw
    measurements (not a 0-100 score) so callers can rank/filter on whichever
    dimension matters most — this only gates whether the pattern is present."""
    min_bars = max(EMA200_TREND_LOOKBACK_DAYS, CONSOLIDATION_LOOKBACK_DAYS) + 1
    if df is None or len(df) < min_bars:
        return {"detected": False, "reason": "insufficient_data"}

    close = df["Close"]
    ema200 = df["EMA200"]
    current_close = float(close.iloc[-1])
    current_ema200 = float(ema200.iloc[-1])
    if pd.isna(current_ema200) or current_ema200 <= 0:
        return {"detected": False, "reason": "insufficient_data"}

    ema200_then = float(ema200.iloc[-1 - EMA200_TREND_LOOKBACK_DAYS])
    if pd.isna(ema200_then) or ema200_then <= 0:
        return {"detected": False, "reason": "insufficient_data"}

    ema200_uptrend_pct = round((current_ema200 - ema200_then) / ema200_then * 100, 2)
    if ema200_uptrend_pct < EMA200_MIN_UPTREND_PCT:
        return {
            "detected": False, "reason": "no_long_term_uptrend",
            "ema200_uptrend_pct": ema200_uptrend_pct,
        }

    price_vs_ema200_pct = round((current_close - current_ema200) / current_ema200 * 100, 2)
    if not (PRICE_VS_EMA200_MIN_PCT <= price_vs_ema200_pct <= PRICE_VS_EMA200_MAX_PCT):
        return {
            "detected": False, "reason": "price_too_far_from_ema200",
            "ema200_uptrend_pct": ema200_uptrend_pct, "price_vs_ema200_pct": price_vs_ema200_pct,
        }

    window = close.tail(CONSOLIDATION_LOOKBACK_DAYS)
    window_low = float(window.min())
    window_high = float(window.max())
    consolidation_range_pct = round((window_high - window_low) / current_close * 100, 2)
    if consolidation_range_pct > CONSOLIDATION_MAX_RANGE_PCT:
        return {
            "detected": False, "reason": "not_consolidating",
            "ema200_uptrend_pct": ema200_uptrend_pct, "price_vs_ema200_pct": price_vs_ema200_pct,
            "consolidation_range_pct": consolidation_range_pct,
        }

    bounce_off_low_pct = round((current_close - window_low) / window_low * 100, 2) if window_low > 0 else 0.0
    if bounce_off_low_pct < MIN_BOUNCE_OFF_LOW_PCT:
        return {
            "detected": False, "reason": "no_reversal_yet",
            "ema200_uptrend_pct": ema200_uptrend_pct, "price_vs_ema200_pct": price_vs_ema200_pct,
            "consolidation_range_pct": consolidation_range_pct, "bounce_off_low_pct": bounce_off_low_pct,
        }

    vp = compute_volume_profile(df, window=VOLUME_PROFILE_WINDOW_DAYS)
    if vp is None:
        return {
            "detected": False, "reason": "insufficient_data",
            "ema200_uptrend_pct": ema200_uptrend_pct, "price_vs_ema200_pct": price_vs_ema200_pct,
            "consolidation_range_pct": consolidation_range_pct, "bounce_off_low_pct": bounce_off_low_pct,
        }
    if current_close > vp["value_area_high"]:
        return {
            "detected": False, "reason": "extended_above_value_area",
            "ema200_uptrend_pct": ema200_uptrend_pct, "price_vs_ema200_pct": price_vs_ema200_pct,
            "consolidation_range_pct": consolidation_range_pct, "bounce_off_low_pct": bounce_off_low_pct,
            "poc": vp["poc"],
        }

    price_vs_poc_pct = round((current_close - vp["poc"]) / vp["poc"] * 100, 2) if vp["poc"] else None
    return {
        "detected": True,
        "ema200_uptrend_pct": ema200_uptrend_pct,
        "price_vs_ema200_pct": price_vs_ema200_pct,
        "consolidation_range_pct": consolidation_range_pct,
        "bounce_off_low_pct": bounce_off_low_pct,
        "poc": vp["poc"],
        "price_vs_poc_pct": price_vs_poc_pct,
    }
