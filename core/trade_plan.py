"""
Trade plan (stop / target / R:R) — ported from swing-finder-v2, NOT from the
unused `trade_plan_levels()` stub in its scanner.py (that function is defined
but never actually called anywhere in the reference app). The real, live path
is the inline stop/target logic in `evaluate_ticker()` (scanner.py:354-511)
plus `utils/target_calculator.py::calculate_fibonacci_target()` and
`utils/indicators.py::find_support_resistance()` — this module ports those.

Stop: swing-low/EMA-anchored, not a flat ATR multiple.
    base_stop = min(10-day swing low, EMA20 - 1.3*ATR14)
    falls back to (price - 1.2*ATR14) if that isn't below price
    then tightened to nearest support cluster if one exists within 3*ATR
Target: Fibonacci 1.618 extension of the most recent 20-bar swing, floored
    at `min_rr_ratio` (settings.min_risk_reward) if the raw extension doesn't
    clear it; refined to the nearest resistance cluster if that's both closer
    and still clears the R:R floor.

Fix vs the reference: `find_support_resistance()`'s nearest-support pick used
index [-1] on a descending-sorted list, i.e. the *farthest* of the top
`num_levels` supports despite the "closest support below price" comment —
this port uses index [0], the actual nearest one.
"""

from __future__ import annotations

import pandas as pd

# Very high R:R (per the tuning sheet's "Advanced Tuning" note) more often means
# an unusually tight stop than an unusually good target — surfaced as a flag
# rather than acted on automatically.
STOP_SANITY_RR_THRESHOLD = 15.0


def find_support_resistance(df: pd.DataFrame, window: int = 10, num_levels: int = 2) -> dict:
    """Cluster pivot highs/lows into support/resistance levels with a touch-count
    strength score. Returns levels below (support) / above (resistance) the
    current price only, closest first."""
    if len(df) < window * 2:
        return {"support": [], "resistance": []}

    highs = df["High"].rolling(window=window, center=True).max()
    lows = df["Low"].rolling(window=window, center=True).min()

    resistance_levels = []
    support_levels = []
    for i in range(window, len(df) - window):
        if df["High"].iloc[i] == highs.iloc[i]:
            level = float(df["High"].iloc[i])
            touches = int((abs(df["High"] - level) / level < 0.01).sum())
            resistance_levels.append({"price": level, "touches": touches})
        if df["Low"].iloc[i] == lows.iloc[i]:
            level = float(df["Low"].iloc[i])
            touches = int((abs(df["Low"] - level) / level < 0.01).sum())
            support_levels.append({"price": level, "touches": touches})

    def cluster(levels: list[dict], tolerance: float = 0.02) -> list[dict]:
        if not levels:
            return []
        levels = sorted(levels, key=lambda x: x["price"])
        clusters, current = [], [levels[0]]
        for level in levels[1:]:
            if abs(level["price"] - current[-1]["price"]) / current[-1]["price"] < tolerance:
                current.append(level)
            else:
                clusters.append(current)
                current = [level]
        clusters.append(current)
        return [
            {"price": sum(l["price"] for l in c) / len(c), "touches": sum(l["touches"] for l in c)}
            for c in clusters
        ]

    resistance_clusters = cluster(resistance_levels)
    support_clusters = cluster(support_levels)

    current_price = float(df["Close"].iloc[-1])

    resistance_above = sorted(
        [r for r in resistance_clusters if r["price"] > current_price], key=lambda x: x["price"]
    )[:num_levels]
    # Closest-first (descending price, since these are all below current_price).
    support_below = sorted(
        [s for s in support_clusters if s["price"] < current_price], key=lambda x: x["price"], reverse=True
    )[:num_levels]

    return {
        "resistance": [round(r["price"], 2) for r in resistance_above],
        "support": [round(s["price"], 2) for s in support_below],
    }


def calculate_fibonacci_target(
    df: pd.DataFrame,
    entry_price: float,
    stop_loss: float,
    lookback_bars: int = 20,
    min_rr_ratio: float = 3.0,
) -> dict:
    """Fibonacci 1.618 extension of the most recent `lookback_bars`-bar swing,
    floored at `min_rr_ratio` if the raw extension falls short. No ceiling —
    a strong extension is allowed to run."""
    window = df.tail(max(lookback_bars, min(15, len(df))))
    swing_high = float(window["High"].max())
    swing_low = float(window["Low"].min())
    swing_range = swing_high - swing_low

    fib_target = swing_high + (1.618 * swing_range)
    risk = abs(entry_price - stop_loss)
    fib_reward = abs(fib_target - entry_price)
    fib_rr = fib_reward / risk if risk > 0 else 0.0
    min_target = entry_price + (min_rr_ratio * risk)

    if fib_rr < min_rr_ratio:
        final_target, final_rr = min_target, min_rr_ratio
        warning = f"Fib extension only {fib_rr:.1f}:1 - using {min_rr_ratio:.0f}:1 floor"
    else:
        final_target, final_rr, warning = fib_target, fib_rr, ""

    return {
        "fib_target": round(fib_target, 2),
        "fib_rr": round(fib_rr, 2),
        "final_target": round(final_target, 2),
        "final_rr": round(final_rr, 2),
        "warning": warning,
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
    }


def resolve_trade_plan_outcome(
    after_bars: pd.DataFrame, stop: float, target: float, max_hold_days: int
) -> tuple[str | None, float | None, "pd.Timestamp | None", int | None]:
    """Walk forward through `after_bars` (bars strictly after the entry point, already
    chronologically sorted), checking each bar's High/Low against stop/target, up to
    max_hold_days bars. Returns (outcome, outcome_price, outcome_date, bars_to_resolution).

    outcome is "target_hit", "stop_hit", "expired_unresolved", or None if after_bars doesn't
    yet span max_hold_days and neither level has been touched — a still-open pick, not yet
    resolvable one way or the other (only meaningful for live tracking of open picks;
    historical backtesting either has enough future bars or excludes the row entirely).

    Same-bar ambiguity (a bar's range touches both stop and target) can't be sequenced from
    daily OHLC data alone — resolved conservatively toward stop_hit (checked first), since
    assuming the better outcome would overstate accuracy. Shared by core.pick_tracking (live
    pick resolution) and research/triple_barrier_walk_forward.py (historical label
    generation) so both use the identical definition of "did this trade work.\""""
    for i in range(min(len(after_bars), max_hold_days)):
        bar = after_bars.iloc[i]
        if bar["Low"] <= stop:
            return "stop_hit", stop, bar["Date"], i + 1
        if bar["High"] >= target:
            return "target_hit", target, bar["Date"], i + 1
    if len(after_bars) >= max_hold_days:
        last_bar = after_bars.iloc[max_hold_days - 1]
        return "expired_unresolved", float(last_bar["Close"]), last_bar["Date"], max_hold_days
    return None, None, None, None


def compute_trade_plan(df: pd.DataFrame, settings) -> dict:
    """Full stop/target/R:R for the most recent bar of `df` (must already have
    compute_indicators() applied). Returns None if there isn't enough data."""
    if df is None or len(df) < 20:
        return None

    last = df.iloc[-1]
    px = float(last["Close"])
    ema20 = float(last["EMA20"])
    atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else px * 0.01

    # --- Base stop: swing-low / EMA-anchored, not a flat ATR multiple ---
    swing_low_10d = float(df["Low"].tail(10).min())
    ema_stop = ema20 - 1.3 * atr_val
    proposed_stop = min(swing_low_10d, ema_stop)
    if proposed_stop >= px:
        proposed_stop = px - 1.2 * atr_val
    stop = max(0.01, proposed_stop)

    # --- Base target: Fibonacci extension floored at settings.min_risk_reward ---
    fib = calculate_fibonacci_target(df, px, stop, lookback_bars=20, min_rr_ratio=settings.min_risk_reward)
    target = fib["final_target"]

    # --- Refine against real support/resistance ---
    sr = find_support_resistance(df, window=10, num_levels=2)

    actual_stop = stop
    if sr["support"]:
        nearest_support = sr["support"][0]
        if px - nearest_support < atr_val * 3:
            actual_stop = nearest_support * 0.995

    actual_target = target
    actual_risk = abs(px - actual_stop)
    if sr["resistance"] and actual_risk > 0:
        nearest_resistance = sr["resistance"][0]
        resistance_target = nearest_resistance * 0.99
        resistance_reward = abs(resistance_target - px)
        resistance_rr = resistance_reward / actual_risk
        if nearest_resistance < target and resistance_rr >= settings.min_risk_reward:
            actual_target = resistance_target

    actual_reward = abs(actual_target - px)
    if actual_risk > 0:
        rr_ratio = round(actual_reward / actual_risk, 2)
        weak_rr = rr_ratio < settings.min_risk_reward
    else:
        rr_ratio = 0.0
        weak_rr = True

    return {
        "entry": round(px, 2),
        "stop": round(actual_stop, 2),
        "target": round(actual_target, 2),
        "rr_ratio": rr_ratio,
        "weak_rr": weak_rr,
        "stop_distance_sanity_flag": rr_ratio >= STOP_SANITY_RR_THRESHOLD,
        "fib_warning": fib["warning"],
    }
