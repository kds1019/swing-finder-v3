"""
Is the fib-extension target too ambitious for this setup?

Uses the path stats (mfe_r, mae_r, won_{k}r) added to the calibration dataset to
compare the CURRENT exit (fib 1.618 extension, floored 3:1 / capped 5:1) against
fixed R-multiple targets, on the calibrated deep-pullback rows.

For a fixed target at kR, each trade resolves as:
    won_{k}r == True                 -> +k
    else, fib outcome == stop_hit    -> -1   (stop was hit before kR)
    else (expired)                   -> mark-to-market close in R (the fib r_multiple,
                                        which for an expired trade IS the 30-bar close)

Usage:  python -m research.exit_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.pullback_reversal import (
    CONSOLIDATION_MAX_RANGE_PCT, EMA200_MIN_UPTREND_PCT,
    MAX_PRICE_VS_VALUE_AREA_HIGH_PCT, PRICE_VS_EMA200_MAX_PCT, PRICE_VS_EMA200_MIN_PCT,
)

DATASET = Path(__file__).resolve().parent / "data" / "calibration_dataset.csv"
KS = (1.0, 1.5, 2.0, 2.5, 3.0)


def _pf(r: pd.Series) -> float:
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return pos / neg if neg > 0 else np.inf


def fixed_target_r(df: pd.DataFrame, k: float) -> pd.Series:
    won = df[f"won_{k}r"].astype("boolean")
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[won.fillna(False)] = k
    lost = ~won.fillna(False)
    stop = lost & (df["outcome"] == "stop_hit")
    out[stop] = -1.0
    exp = lost & (df["outcome"] != "stop_hit")
    out[exp] = df.loc[exp, "r_multiple"]        # expired -> mark-to-market close in R
    return out


def main() -> None:
    if not DATASET.exists():
        sys.exit("run research.build_calibration_dataset first")
    d = pd.read_csv(DATASET, parse_dates=["date"], low_memory=False)
    if "mfe_r" not in d.columns:
        sys.exit("dataset has no path stats — rebuild with the updated build_calibration_dataset.py")

    d = d[~d["weak_rr"].astype(bool)].copy()
    deep = d[
        (d.ema200_uptrend_pct >= EMA200_MIN_UPTREND_PCT)
        & d.price_vs_ema200_pct.between(PRICE_VS_EMA200_MIN_PCT, PRICE_VS_EMA200_MAX_PCT)
        & (d.consolidation_range_pct <= CONSOLIDATION_MAX_RANGE_PCT)
        & (d.price_vs_value_area_high_pct <= MAX_PRICE_VS_VALUE_AREA_HIGH_PCT)
    ].copy()

    print(f"calibrated deep-pullback rows: {len(deep):,}\n")

    # --- MFE: how far trades actually run before resolving ---
    mfe = deep["mfe_r"].dropna()
    print("Max favourable excursion (R), over the hold window:")
    print(f"  median {mfe.median():.2f}   mean {mfe.mean():.2f}")
    for k in KS:
        print(f"  reached +{k}R at some point: {(mfe >= k).mean()*100:5.1f}%")
    print(f"  of trades that STOPPED OUT on the fib target, "
          f"{(deep.loc[deep.outcome=='stop_hit','mfe_r'] >= 2).mean()*100:.1f}% had first touched +2R")
    print()

    # --- exit comparison ---
    cur = deep["r_multiple"]
    hdr = f"{'exit rule':<22}{'win%':>7}{'avg_R':>9}{'med_R':>8}{'PF':>7}"
    print(hdr); print("-" * len(hdr))
    print(f"{'fib ext (current)':<22}{(deep.outcome=='target_hit').mean()*100:>6.1f}%"
          f"{cur.mean():>+9.3f}{cur.median():>+8.2f}{_pf(cur):>7.2f}")
    for k in KS:
        r = fixed_target_r(deep, k)
        print(f"{'fixed +' + str(k) + 'R':<22}{(r == k).mean()*100:>6.1f}%"
              f"{r.mean():>+9.3f}{r.median():>+8.2f}{_pf(r):>7.2f}")

    # --- by year, best fixed target vs fib ---
    print("\nby year (avg_R):   fib   +1R   +1.5R  +2R   +2.5R  +3R")
    for y, g in deep.groupby(deep.date.dt.year):
        vals = [g['r_multiple'].mean()] + [fixed_target_r(g, k).mean() for k in KS]
        print(f"  {y}   " + "  ".join(f"{v:+.2f}" for v in vals))


if __name__ == "__main__":
    main()
