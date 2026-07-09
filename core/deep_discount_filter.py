"""
Deep Discount Stabilization Filter — FRESH DESIGN, not a port.

swing-finder-v2 has no function or logic with this name anywhere in the codebase
(confirmed via exhaustive search) — only the Fibonacci "Deep Discount (0-23.6%)"
zone label and its associated SmartScore bonus exist. This module implements a
reasonable interpretation of what a "stabilization filter" on top of that zone
should do: avoid rewarding a deep-discount Fibonacci reading that is really just
a falling knife still in motion, by requiring evidence the price has stopped
making fresh lows and volatility/volume are calming down.

Validate this against the actual Google Sheet definition before trusting it at
full scale — there is no prior implementation to verify parity against.
"""

from __future__ import annotations

import pandas as pd

DEEP_DISCOUNT_FIB_THRESHOLD = 23.6  # matches indicators.get_fibonacci_zone_label's "Deep Discount" bucket
STABILIZATION_CHECKS_REQUIRED = 2   # out of 3
STABILIZATION_PENALTY = -20
ATR_COMPRESSION_MULTIPLE = 1.5
VOLUME_SPIKE_THRESHOLD = 2.0


def evaluate_deep_discount_stabilization(df: pd.DataFrame, fib_data: dict | None, rel_vol: float) -> dict:
    """
    Only engages when fib_data indicates a "discount" zone at or below the Deep
    Discount threshold (fib_position <= 23.6). Otherwise a no-op.

    Requires >= STABILIZATION_CHECKS_REQUIRED of 3 checks to pass:
      1. Recent 3-bar average true range <= ATR_COMPRESSION_MULTIPLE x ATR14
         (volatility calming down, not still crashing).
      2. No new 20-day low in the last 3 bars (price isn't still making fresh lows).
      3. rel_vol < VOLUME_SPIKE_THRESHOLD (no active panic/capitulation volume spike).

    If fewer than the required checks pass, returns a SmartScore penalty and a
    flag rather than a hard exclude, so the ticker stays visible for review.
    """
    if not fib_data or fib_data.get("zone") != "discount" or fib_data.get("current_fib_position", 100) > DEEP_DISCOUNT_FIB_THRESHOLD:
        return {"triggered": False, "checks_passed": None, "penalty": 0, "flag": None}

    if len(df) < 21 or "ATR14" not in df.columns or "LL20" not in df.columns:
        return {"triggered": True, "checks_passed": 0, "penalty": STABILIZATION_PENALTY, "flag": "unstable_deep_discount_insufficient_data"}

    last3 = df.tail(3)
    atr14 = float(df["ATR14"].iloc[-1])

    # Check 1: volatility compression over the last 3 bars vs ATR14
    true_ranges = (last3["High"] - last3["Low"]).abs()
    check_atr_calm = bool(true_ranges.mean() <= ATR_COMPRESSION_MULTIPLE * atr14) if atr14 > 0 else False

    # Check 2: no new 20-day low within the last 3 bars
    ll20_before = float(df["LL20"].iloc[-4]) if len(df) >= 4 else float(df["LL20"].iloc[-1])
    check_no_new_low = bool(last3["Low"].min() >= ll20_before)

    # Check 3: not in an active volume-spike capitulation
    check_volume_calm = bool(rel_vol < VOLUME_SPIKE_THRESHOLD)

    checks_passed = sum([check_atr_calm, check_no_new_low, check_volume_calm])

    if checks_passed >= STABILIZATION_CHECKS_REQUIRED:
        return {"triggered": True, "checks_passed": checks_passed, "penalty": 0, "flag": None}

    return {
        "triggered": True,
        "checks_passed": checks_passed,
        "penalty": STABILIZATION_PENALTY,
        "flag": "unstable_deep_discount",
    }
