"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Standalone isolated backtest of a DIFFERENT entry pattern than the live screener's, tested
per direct request without changing anything live: a stock whose 200-day EMA is genuinely
uptrending, that has pulled back through its 50-day EMA, but has stabilised (same
stabilization definition used everywhere else in this codebase —
core.pullback_reversal.classify_knife_risk == "stabilising") while still holding ABOVE its
200-day EMA — the classic "pullback to the 50, holding the 200, in an uptrend" shape.

This is a genuinely different candidate population from the live screener's own pattern
(which requires price within -20%/+3% of EMA200 itself — much closer to or below the
200-day), not a threshold variant of the existing gate — so it stands alone here rather than
comparing variants of one signal the way g1_ab.py / current_trend_gate_ab.py do.

Reuses the same portfolio engine/costs as every other isolated test in this repo
(research/current_trend_gate_ab.py's run()/stat()) and the same trade-management machinery
(core/trade_plan.py's stop/target/trailing exit, weak-RR dropped) — holding trade management
constant isolates the ENTRY PATTERN as the one thing actually being tested.

Uptrend definition reuses the already-validated live constants rather than re-deriving new
ones: EMA200_MIN_UPTREND_PCT (5% over EMA200_TREND_LOOKBACK_DAYS, 126 sessions) for the
long-horizon trend, PLUS EMA200_CURRENT_SLOPE_MIN_PCT (-2% over 20 sessions, the gate added
2026-09-11) so the "uptrend" read isn't stale — both are the live screener's own gate.

Usage:  python -m research.pullback_to_50_ema_ab            (uses cache if present)
        python -m research.pullback_to_50_ema_ab --rebuild
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
    MIN_BARS_FOR_SCREENER, EMA200_TREND_LOOKBACK_DAYS, EMA200_MIN_UPTREND_PCT,
    EMA200_CURRENT_SLOPE_LOOKBACK_DAYS, EMA200_CURRENT_SLOPE_MIN_PCT,
    measure_stabilization,
)
from core.trade_plan import compute_trade_plan
from core.universe import build_universe
from research.current_trend_gate_ab import run, stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "pullback_to_50_signals.pkl"
OUT = Path(__file__).resolve().parent / "pullback_to_50_ema_ab.md"

START = "2021-06-01"
WINDOW = 300


def build_signals(settings) -> pd.DataFrame:
    uni = build_universe(settings)
    sectors = dict(zip(uni["Ticker"], uni["Sector"]))
    tickers = sorted(p.stem for p in BARS_DIR.glob("*.pkl")
                      if p.stem != "SPY" and p.stem in sectors)
    print(f"[pb50] {len(tickers)} tickers", file=sys.stderr)
    sig = []
    for n, t in enumerate(tickers, 1):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        close, ema50, ema200 = df["Close"], df["EMA50"], df["EMA200"]
        upt = (ema200 / ema200.shift(EMA200_TREND_LOOKBACK_DAYS) - 1.0) * 100.0
        cur_slope = (ema200 / ema200.shift(EMA200_CURRENT_SLOPE_LOOKBACK_DAYS) - 1.0) * 100.0
        below_50 = close < ema50
        above_200 = close > ema200
        net = (
            (upt >= EMA200_MIN_UPTREND_PCT) & (cur_slope >= EMA200_CURRENT_SLOPE_MIN_PCT)
            & below_50 & above_200
        )
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1:
                continue
            d = df["Date"].iloc[i]
            if str(d.date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            stab = measure_stabilization(prefix)
            if stab.get("knife_risk_tier") != "stabilising":
                continue
            try:
                plan = compute_trade_plan(prefix, settings)
            except ZeroDivisionError:
                # core.trade_plan.find_support_resistance.cluster() can divide by a ~0 pivot
                # price on deep-split-adjusted ancient bars — same known issue as g1_ab.py.
                continue
            if plan is None or plan["stop"] >= plan["entry"] or plan["weak_rr"]:
                continue
            px = float(close.iloc[i])
            sig.append({
                "date": d, "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "depth": round((px / float(ema200.iloc[i]) - 1.0) * 100.0, 2),
                "tier": "stabilising",
                "price_vs_ema50_pct": round((px / float(ema50.iloc[i]) - 1.0) * 100.0, 2),
            })
        if n % 100 == 0:
            print(f"[pb50] {n}/{len(tickers)} tickers, {len(sig)} signals so far", file=sys.stderr)
    return pd.DataFrame(sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if SIG_CACHE.exists() and not a.rebuild:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[pb50] loaded {len(sig)} cached signals", file=sys.stderr)
    else:
        sig = build_signals(settings)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[pb50] built + cached {len(sig)} signals", file=sys.stderr)

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    if sig.empty:
        print("[pb50] zero signals matched this pattern — nothing to backtest", file=sys.stderr)
        return

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    res = run(sig, bars, spy_above, calendar, settings)

    L = [
        "# Pullback-to-50-EMA (holding the 200-EMA) — standalone isolated backtest (RESEARCH)", "",
        "A DIFFERENT entry pattern from the live screener's own -20%/+3%-vs-EMA200 deep "
        "pullback — tested standalone, not a threshold variant, per direct request without "
        "touching anything live. Criteria: EMA200 genuinely uptrending (126-day slope >=5%, "
        "the live G1 threshold) AND not stale (20-day slope >=-2%, the live current-trend "
        "gate) AND price below EMA50 AND price above EMA200 AND the same stabilization "
        "signal used everywhere else in this codebase (KnifeRiskTier == \"stabilising\"). "
        "Same trade management as the live system (swing-low/EMA-anchored stop, "
        "Fibonacci-extension target, +2R/give-1R trailing exit, weak-RR dropped) — holding "
        "that constant isolates the entry pattern as what's actually being tested.",
        f"- {len(sig)} signals matched, {sig['ticker'].nunique()} tickers, {START} .. present",
        "- engine: research/current_trend_gate_ab.py's run()/stat() — same costs/sizing/"
        "sector cap as every other isolated test in this repo", "",
    ]

    for wn, lo, hi in windows:
        L += [f"## {wn}", "",
              "| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        s = stat(res, spy, lo, hi)
        if s is None:
            L.append("| Pullback-to-50 | (insufficient) | | | | | | | |")
        else:
            L.append(f"| Pullback-to-50 | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | "
                      f"{s['sharpe']:.2f} | {s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | "
                      f"{s['pf']:.2f} | {s['mcl']} |")
            L.append(f"| SPY | {s['spy_ret']*100:+.0f}% | | | | | | | |")
        L.append("")

    L += ["_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not "
          "corrected. This is a NEW, untested-until-now pattern — treat these numbers with the "
          "same skepticism as any first backtest, not as validated the way the live screener's "
          "own thresholds are (those went through calibration + train/test + this same "
          "isolated-portfolio check before being trusted)._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[pb50] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
