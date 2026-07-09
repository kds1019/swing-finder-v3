"""
Volume profile position scoring — FRESH DESIGN, not a port.

swing-finder-v2 has no volume-profile/point-of-control concept anywhere in its
codebase. This bins the ticker's own already-fetched OHLCV bars into a
price-by-volume histogram and folds "is price at/below its point of control
(real volume support underneath) or stretched above it (thin trading, less
support)" into the SmartScore as a bonus/penalty — same post-hoc-adjustment
pattern as core/deep_discount_filter.py.

Deliberately cheap (pure binning, no model training) so it can run on the full
universe scan, not just the post-sector-cap shortlist — unlike the ML-edge
adjustment in core/ml_forecast.py, which needs per-ticker model training and
is therefore shortlist-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUM_BINS = 20
AT_OR_BELOW_POC_BONUS = 8
ABOVE_POC_PENALTY = -12
EXTENDED_ABOVE_VALUE_AREA_PENALTY = -18  # past where 70% of recent volume actually traded


def compute_volume_profile(df: pd.DataFrame, window: int | None = None, num_bins: int = NUM_BINS) -> dict | None:
    """Price-by-volume histogram over the trailing `window` bars (default: all
    bars the caller passed in). Returns None if there isn't enough data to
    bin meaningfully or the price range is degenerate."""
    vp_bars = df.tail(window) if window else df
    if len(vp_bars) < 10:
        return None

    lo, hi = float(vp_bars["Low"].min()), float(vp_bars["High"].max())
    if hi <= lo:
        return None

    bins = np.linspace(lo, hi, num_bins + 1)
    vol_by_bin = np.zeros(num_bins)

    for _, row in vp_bars.iterrows():
        row_lo, row_hi, vol = row["Low"], row["High"], row["Volume"]
        if row_hi <= row_lo or vol <= 0:
            idx = int(np.clip(np.searchsorted(bins, row["Close"]) - 1, 0, num_bins - 1))
            vol_by_bin[idx] += vol
            continue
        overlap_lo = np.maximum(bins[:-1], row_lo)
        overlap_hi = np.minimum(bins[1:], row_hi)
        overlap = np.clip(overlap_hi - overlap_lo, 0, None)
        total_overlap = overlap.sum()
        weights = overlap / total_overlap if total_overlap > 0 else np.zeros(num_bins)
        vol_by_bin += weights * vol

    poc_idx = int(np.argmax(vol_by_bin))
    poc = round(float((bins[poc_idx] + bins[poc_idx + 1]) / 2), 2)

    total_vol = vol_by_bin.sum()
    order = np.argsort(vol_by_bin)[::-1]
    cum, va_bins = 0.0, []
    for idx in order:
        va_bins.append(idx)
        cum += vol_by_bin[idx]
        if cum >= 0.70 * total_vol:
            break

    return {
        "poc": poc,
        "value_area_low": round(float(bins[min(va_bins)]), 2),
        "value_area_high": round(float(bins[max(va_bins) + 1]), 2),
    }


def evaluate_volume_profile_position(df: pd.DataFrame, window: int | None = None) -> dict:
    """SmartScore adjustment for where price sits relative to its own volume
    profile: at/below the point of control (real volume support underneath)
    is rewarded; stretched above it - especially past the value area where
    70% of recent volume traded - is penalized."""
    vp = compute_volume_profile(df, window=window)
    if vp is None:
        return {"triggered": False, "score_adjustment": 0, "flag": None, "poc": None, "price_vs_poc_pct": None}

    price = float(df["Close"].iloc[-1])
    poc = vp["poc"]
    price_vs_poc_pct = round((price - poc) / poc * 100, 2) if poc else None

    if price <= poc:
        return {
            "triggered": True, "score_adjustment": AT_OR_BELOW_POC_BONUS,
            "flag": None, "poc": poc, "price_vs_poc_pct": price_vs_poc_pct,
        }

    if price > vp["value_area_high"]:
        return {
            "triggered": True, "score_adjustment": EXTENDED_ABOVE_VALUE_AREA_PENALTY,
            "flag": "extended_above_value_area", "poc": poc, "price_vs_poc_pct": price_vs_poc_pct,
        }

    return {
        "triggered": True, "score_adjustment": ABOVE_POC_PENALTY,
        "flag": "above_poc", "poc": poc, "price_vs_poc_pct": price_vs_poc_pct,
    }
