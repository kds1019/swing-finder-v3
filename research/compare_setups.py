"""
Is the deep-pullback setup a real edge, or just bull-market beta?

The decisive test: does entering on the setup beat entering on a RANDOM day in the
same stocks over the same period, with the same stop/target and horizon?

  - any_dip / deep_pb  : taken straight from research/data/calibration_dataset.csv
      any_dip = every wide-net row (rising 200-EMA + within -22%..+14% of it)
      deep_pb = rows passing the calibrated core.pullback_reversal gates
  - random             : computed fresh here — N random bars per cached ticker,
      same _label_at() (compute_trade_plan + resolve_trade_plan_outcome, 30-bar
      horizon) as build_calibration_dataset.py. The "no timing skill" baseline.

Also prints buy-and-hold of the same tickers over the labelled window.

Usage:
    python -m research.compare_setups --per-ticker-random 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    CONSOLIDATION_MAX_RANGE_PCT,
    EMA200_MIN_UPTREND_PCT,
    MAX_PRICE_VS_VALUE_AREA_HIGH_PCT,
    MIN_BARS_FOR_SCREENER,
    PRICE_VS_EMA200_MAX_PCT,
    PRICE_VS_EMA200_MIN_PCT,
)
from core.trade_plan import compute_trade_plan, resolve_trade_plan_outcome

MAX_HOLD_DAYS = 30
WINDOW_BARS = 300
BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
DATASET = Path(__file__).resolve().parent / "data" / "calibration_dataset.csv"


def _label_at(df: pd.DataFrame, i: int, settings) -> dict | None:
    prefix = df.iloc[: i + 1].tail(WINDOW_BARS)
    try:
        plan = compute_trade_plan(prefix, settings)
    except ZeroDivisionError:
        # core.trade_plan.find_support_resistance.cluster() divides by a pivot price that
        # can be ~0 on deep-split-adjusted ancient bars — a random bar can land there.
        # Latent production bug (only dodged because live scans are all >= $10); skip here.
        return None
    if plan is None:
        return None
    after = df.iloc[i + 1 : i + 1 + MAX_HOLD_DAYS][["Date", "High", "Low", "Close"]].reset_index(drop=True)
    outcome, px, _, bars_to = resolve_trade_plan_outcome(after, plan["stop"], plan["target"], MAX_HOLD_DAYS)
    if outcome is None:
        return None
    risk = plan["entry"] - plan["stop"]
    return {"outcome": outcome, "r_multiple": (px - plan["entry"]) / risk if risk > 0 else np.nan,
            "weak_rr": plan["weak_rr"], "bars_to": bars_to}


def _metrics(df: pd.DataFrame, drop_weak: bool = True) -> str:
    if drop_weak and "weak_rr" in df:
        df = df[~df["weak_rr"].astype(bool)]
    r = df["r_multiple"].replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return "(no rows)"
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    hold = df["bars_to"].mean() if "bars_to" in df else df["bars_to_resolution"].mean()
    return (f"n={len(df):>7}  hit={(df['outcome']=='target_hit').mean()*100:>4.1f}%  "
            f"avgR={r.mean():>+6.3f}  medR={r.median():>+5.2f}  PF={pos/neg:>4.2f}  hold={hold:>4.1f}d")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-ticker-random", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    settings = load_settings()
    rng = np.random.default_rng(args.seed)

    if not DATASET.exists():
        sys.exit("run research.build_calibration_dataset first")
    d = pd.read_csv(DATASET, parse_dates=["date"])

    any_dip = d
    deep_pb = d[
        (d.ema200_uptrend_pct >= EMA200_MIN_UPTREND_PCT)
        & d.price_vs_ema200_pct.between(PRICE_VS_EMA200_MIN_PCT, PRICE_VS_EMA200_MAX_PCT)
        & (d.consolidation_range_pct <= CONSOLIDATION_MAX_RANGE_PCT)
        & (d.price_vs_value_area_high_pct <= MAX_PRICE_VS_VALUE_AREA_HIGH_PCT)
    ]

    ds_tickers = set(d["ticker"].unique())
    caches = sorted(x for x in BARS_DIR.glob("*.pkl") if x.stem != "SPY" and x.stem in ds_tickers)

    rand_rows: list[dict] = []
    bh: list[float] = []
    for n, cache in enumerate(caches, 1):
        raw = pd.read_pickle(cache)
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + MAX_HOLD_DAYS + 5:
            continue
        df = compute_indicators(raw.copy())
        first_ok = max(MIN_BARS_FOR_SCREENER - 1, 300)
        last_ok = len(df) - MAX_HOLD_DAYS - 1
        if last_ok <= first_ok:
            continue
        idx = np.arange(first_ok, last_ok + 1)
        bh.append(df["Close"].iloc[last_ok] / df["Close"].iloc[first_ok] - 1.0)
        for i in rng.choice(idx, size=min(args.per_ticker_random, len(idx)), replace=False):
            r = _label_at(df, int(i), settings)
            if r:
                rand_rows.append(r)
        if n % 100 == 0:
            print(f"[compare] {n}/{len(caches)} tickers  ({len(rand_rows)} random rows)", file=sys.stderr)

    rand = pd.DataFrame(rand_rows)

    print(f"\nbuy & hold, same tickers, labelled window: mean {np.mean(bh)*100:+.1f}%  median {np.median(bh)*100:+.1f}%\n")
    print(f"{'random entry ':<22}{_metrics(rand)}")
    print(f"{'any dip in uptrend ':<22}{_metrics(any_dip)}")
    print(f"{'calibrated deep_pb ':<22}{_metrics(deep_pb)}")

    rr = rand[~rand['weak_rr'].astype(bool)]['r_multiple']
    dp = deep_pb[~deep_pb['weak_rr'].astype(bool)]['r_multiple']
    ad = any_dip[~any_dip['weak_rr'].astype(bool)]['r_multiple']
    print(f"\ndeep_pb avgR - random avgR  = {dp.mean() - rr.mean():+.3f} R/trade")
    print(f"deep_pb avgR - any_dip avgR = {dp.mean() - ad.mean():+.3f} R/trade")
    print(f"any_dip avgR - random avgR  = {ad.mean() - rr.mean():+.3f} R/trade")


if __name__ == "__main__":
    main()
