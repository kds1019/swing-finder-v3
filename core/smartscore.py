"""
SmartScore engine — ported verbatim (formula and constants) from swing-finder-v2's
scanner.py::evaluate_ticker, restructured as pure functions over a plain OHLCV+indicators
DataFrame instead of being inlined in a Streamlit scan loop.

Note on scope: swing-finder-v2's evaluate_ticker also computes stop/target/R:R via
calculate_scanner_target()/find_support_resistance() (in utils/target_calculator.py
and utils/indicators.py). That engine was out of scope for this module's initial
build but has since been ported — see core/trade_plan.py::compute_trade_plan(),
wired into agents/market_data_agent.py::scan_universe().

Simplification vs the reference: the reference repo has two separate, inconsistent
setup classifiers — a simple top-level `classify_setup()` (ema/rsi only) that isn't
actually used by the live scan path, and a more detailed threshold-based classifier
inlined in `evaluate_ticker` that IS what drives real scoring. This port keeps only
the one that actually matters, as `classify_setup()` below.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from core.indicators import find_pivot_points, calculate_fibonacci_levels

# Sensitivity level 3 ("Balanced") thresholds from swing-finder-v2's sensitivity_map.
# The reference app exposes this as a UI slider (1-5); this headless pipeline fixes it
# at the app's own default (3) since there's no UI to adjust it per-run.
SETUP_THRESHOLDS = {
    "breakout_rsi": 55,
    "breakout_band": 0.55,
    "pullback_rsi_min": 35,
    "pullback_rsi_max": 50,
    "pullback_band": 0.45,
}

# Market-bias buffer magnitudes (SPY EMA20 vs EMA50) — always-on in this pipeline,
# unlike the reference app where it's gated behind an opt-in "Smart Mode" toggle.
MARKET_BIAS_RSI_BUFFER = 3
MARKET_BIAS_BAND_BUFFER = 0.05


def passes_filters(last: pd.Series, settings) -> bool:
    """Price range and min-volume sanity filters."""
    px = float(last["Close"])
    vol = float(last["Volume"])
    if pd.isna(px) or pd.isna(vol):
        return False
    if px < settings.price_min or px > settings.price_max:
        return False
    if vol < settings.min_volume:
        return False
    return True


def apply_market_bias_buffer(thresholds: dict, market_bias: str | None) -> dict:
    """
    Loosen thresholds in an uptrend (catch more opportunities), tighten in a
    downtrend (avoid traps). market_bias is "Uptrend" / "Downtrend" / None,
    from SPY EMA20 vs EMA50 (see pipeline.py's market bias computation).
    """
    rsi_buffer = 0.0
    band_buffer = 0.0
    if market_bias == "Uptrend":
        rsi_buffer = -MARKET_BIAS_RSI_BUFFER
        band_buffer = -MARKET_BIAS_BAND_BUFFER
    elif market_bias == "Downtrend":
        rsi_buffer = MARKET_BIAS_RSI_BUFFER
        band_buffer = MARKET_BIAS_BAND_BUFFER

    return {
        "breakout_rsi": thresholds["breakout_rsi"] + rsi_buffer,
        "breakout_band": thresholds["breakout_band"] + band_buffer,
        "pullback_rsi_min": thresholds["pullback_rsi_min"] + rsi_buffer,
        "pullback_rsi_max": thresholds["pullback_rsi_max"] + rsi_buffer,
        "pullback_band": thresholds["pullback_band"] + band_buffer,
    }


def classify_setup(last: pd.Series, thresholds: dict) -> tuple[str | None, bool, str | None]:
    """
    Returns (setup, near_miss, near_type). setup is "Breakout" / "Pullback" / None.
    near_miss flags tickers close to qualifying but not quite there, for visibility.
    """
    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI14"])
    band = float(last["BandPos20"])
    px = float(last["Close"])
    support = float(last.get("LL20", np.nan))
    resistance = float(last.get("HH20", np.nan))
    atr_val = float(last.get("ATR14", np.nan))
    atr_val = px * 0.01 if pd.isna(atr_val) or atr_val <= 0 else atr_val

    setup, near_miss, near_type = None, False, None

    if ema20 > ema50:
        if rsi > thresholds["breakout_rsi"] and band > thresholds["breakout_band"]:
            setup = "Breakout"
        elif (thresholds["pullback_rsi_min"] <= rsi <= thresholds["pullback_rsi_max"]
                and band <= thresholds["pullback_band"] and px <= ema20):
            setup = "Pullback"

        if not setup:
            near_pct = 15.0
            near_atr_mult = 4.0

            if 40 <= rsi <= 67 and 0.35 <= band <= 0.70:
                near_miss, near_type = True, "RSI/Band breakout proximity"
            elif 40 <= rsi <= 70 and 0.20 <= band <= 0.60:
                near_miss, near_type = True, "RSI/Band pullback proximity"

            if pd.notna(resistance) and resistance > 0 and (resistance - px) / resistance <= near_pct / 100:
                near_miss, near_type = True, f"<={near_pct:.0f}% below 20-day high"
            elif pd.notna(support) and (px - support) <= near_atr_mult * atr_val:
                near_miss, near_type = True, f"<={near_atr_mult:.1f}xATR above 20-day low"

    return setup, near_miss, near_type


def compute_smartscore(df: pd.DataFrame, market_bias: str | None, settings) -> dict:
    """
    Compute the SmartScore for the most recent bar of `df` (must already have
    compute_indicators() applied). Returns a dict with the score, setup
    classification, and a factor-by-factor breakdown for transparency.

    Returns {"smartscore": None, "reason": ...} if there isn't enough data,
    the ticker fails basic filters, or there's no setup/near-miss signal.
    """
    if df is None or len(df) < 60:
        return {"smartscore": None, "setup": None, "reason": "insufficient_data"}

    last = df.iloc[-1]

    if not passes_filters(last, settings):
        return {"smartscore": None, "setup": None, "reason": "filtered"}

    ema20 = float(last["EMA20"])
    ema50 = float(last["EMA50"])
    rsi = float(last["RSI14"])
    band = float(last["BandPos20"])

    thresholds = apply_market_bias_buffer(SETUP_THRESHOLDS, market_bias)
    setup, near_miss, near_type = classify_setup(last, thresholds)

    if not setup and not near_miss:
        return {"smartscore": None, "setup": None, "reason": "no_signal"}

    breakdown: dict[str, float] = {"baseline": float(settings.smartscore_baseline)}
    smart_score = float(settings.smartscore_baseline)

    # --- Setup strength (RSI/band alignment) ---
    if setup == "Breakout":
        setup_strength = min((rsi - 50) * 1.2, 25) + min((band - 0.5) * 50, 15)
    elif setup == "Pullback":
        setup_strength = min((60 - rsi) * 1.2, 25) + min((0.5 - band) * 50, 15)
    else:
        setup_strength = 0.0
    smart_score += setup_strength
    breakdown["setup_strength"] = setup_strength

    # --- Trend context ---
    trend_bonus = 10 if ema20 > ema50 else -10
    smart_score += trend_bonus
    breakdown["trend_context"] = trend_bonus

    # --- Volume ---
    vol_20_avg = df["Volume"].tail(20).mean()
    vol = float(last["Volume"])
    rel_vol = vol / vol_20_avg if vol_20_avg > 0 else 1.0
    if rel_vol >= 1.5:
        vol_bonus = 15
    elif rel_vol >= 1.0:
        vol_bonus = 5
    elif rel_vol < 0.8:
        vol_bonus = -10
    else:
        vol_bonus = 0
    smart_score += vol_bonus
    breakdown["volume"] = vol_bonus

    smart_score = int(np.clip(smart_score, 0, 100))

    # --- Base detection (tight consolidation 15-to-3 bars ago) ---
    has_base = False
    base_tightness = None
    base_bonus = 0
    if len(df) >= 15:
        pre_move = df.iloc[-15:-3]
        base_high = float(pre_move["High"].max())
        base_low = float(pre_move["Low"].min())
        base_mid = float(pre_move["Close"].mean())
        if base_mid > 0:
            base_range = (base_high - base_low) / base_mid
            if base_range < 0.07:
                has_base = True
                base_tightness = round(base_range * 100, 1)
                if setup == "Breakout":
                    base_bonus = 12
    smart_score += base_bonus
    breakdown["base_tightness"] = base_bonus

    # --- Meaningful level (Pullback / near-miss: proximity to EMA20/EMA50/pivot low) ---
    at_meaningful_level = False
    level_description = None
    level_bonus = 0
    if setup == "Pullback" or near_miss:
        px = float(last["Close"])
        ema20_dist = abs(px - ema20) / px if px > 0 else 1
        ema50_dist = abs(px - ema50) / px if px > 0 else 1

        piv = find_pivot_points(df.tail(30), left_bars=3, right_bars=3)
        piv_lows = piv["pivot_lows"]
        near_pivot_low = False
        nearest_pl_price = None
        if piv_lows:
            nearest_pl = min(piv_lows, key=lambda p: abs(p["price"] - px))
            pivot_dist = abs(nearest_pl["price"] - px) / px
            near_pivot_low = pivot_dist <= 0.025
            nearest_pl_price = nearest_pl["price"]

        if ema20_dist <= 0.02:
            at_meaningful_level = True
            level_description = f"near EMA20 (${ema20:.2f})"
            level_bonus = 10
        elif ema50_dist <= 0.02:
            at_meaningful_level = True
            level_description = f"near EMA50 (${ema50:.2f})"
            level_bonus = 8
        elif near_pivot_low and nearest_pl_price:
            at_meaningful_level = True
            level_description = f"near pivot low (${nearest_pl_price:.2f})"
            level_bonus = 8
    smart_score += level_bonus
    breakdown["meaningful_level"] = level_bonus

    smart_score = int(np.clip(smart_score, 0, 100))

    # --- Fibonacci zone ---
    fib_data = calculate_fibonacci_levels(df, lookback=20)
    fib_bonus = 0
    if fib_data:
        fib_position = fib_data["current_fib_position"]
        fib_zone = fib_data["zone"]
        if fib_zone == "discount":
            if fib_position <= 38.2:
                fib_bonus = 15
            elif fib_position <= 50:
                fib_bonus = 10
        elif fib_zone == "premium":
            if fib_position >= 78.6:
                fib_bonus = -15
            elif fib_position >= 61.8:
                fib_bonus = -10
        smart_score = max(0, min(100, round(smart_score + fib_bonus, 1)))
    breakdown["fibonacci"] = fib_bonus

    return {
        "smartscore": smart_score,
        "setup": setup,
        "near_miss": near_miss,
        "near_type": near_type,
        "breakdown": breakdown,
        "rel_vol": round(rel_vol, 2),
        "has_base": has_base,
        "base_tightness": base_tightness,
        "at_meaningful_level": at_meaningful_level,
        "level_description": level_description,
        "fib_data": fib_data,
        "market_bias_thresholds": thresholds,
    }
