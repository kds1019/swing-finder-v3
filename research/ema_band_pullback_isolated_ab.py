"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated portfolio test of the EMA-band-pullback zone (price <= 50-EMA, no worse than -20%
vs 200-EMA, TrendState == uptrend, KnifeRiskTier == stabilising, NOT already
trend_continuation) at the stabilization-duration threshold ACTUALLY LIVE today
(core.trend_context.EMA_BAND_STABILIZATION_MIN_DAYS) — imported directly rather than
hardcoded, so this report can never silently drift out of sync with the live constant. That
threshold started at 10 days (a first-cut guess, validated at the time: 175 trades, positive
every window) and was lowered to 8 after research/ema_band_stabilization_sweep.py found 8
MORE robust, not less — PF 1.39-1.55 in every window vs. 10 days' softer 1.21 in the test
window.

Tested on its OWN dedicated capital — no competition for capacity against reversion_bounce
or any other bucket, unlike the blended-portfolio version where this pattern mostly lost
capacity fights under the portfolio's depth-based ordering (only 11 of 318 raw 10-day-cutoff
signals ever got traded there). This test gives every raw signal a fair shot, the same way
research/pullback_to_50_ema_ab.py and every other standalone pattern test in this repo was
run.

Reuses research/current_trend_gate_ab.py's run()/stat() engine (same costs/sizing/sector cap/
regime filter as every other isolated test here) and
research/ema_band_stabilization_sweep.py's already-cached, already-tagged signal set + zone
check — no new tagging pass needed.

Usage:  python -m research.ema_band_pullback_isolated_ab
"""

from __future__ import annotations

import sys

import pandas as pd

from config.settings import load_settings
from core.trend_context import EMA_BAND_STABILIZATION_MIN_DAYS
from research.current_trend_gate_ab import BARS_DIR, run, stat
from research.ema_band_stabilization_sweep import SIG_CACHE, in_ema_band_zone

OUT_PATH = "research/ema_band_pullback_isolated_ab.md"
START = "2021-06-01"


def main():
    settings = load_settings()
    if not SIG_CACHE.exists():
        sys.exit("research/data/ema_band_v2_signals.pkl not found — run "
                  "`python -m research.ema_band_pullback_v2_ab` first.")
    sig = pd.read_pickle(SIG_CACHE)
    zone = sig[sig.apply(in_ema_band_zone, axis=1)]
    sig = zone[zone["days_since_pullback_low"] >= EMA_BAND_STABILIZATION_MIN_DAYS].reset_index(drop=True)
    print(f"[ema_isolated] {len(sig)} raw signals at days_since_pullback_low >= "
          f"{EMA_BAND_STABILIZATION_MIN_DAYS}, {sig['ticker'].nunique()} tickers", file=sys.stderr)

    if sig.empty:
        sys.exit("No signals in the EMA-band zone at this threshold — nothing to backtest.")

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    res = run(sig, bars, spy_above, calendar, settings)

    L = [f"# EMA-band pullback ({EMA_BAND_STABILIZATION_MIN_DAYS}+ day stabilization) — "
         "standalone isolated backtest (RESEARCH)", "",
         "The live core.trend_context.EMA_BAND_STABILIZATION_MIN_DAYS threshold, tested on "
         "its OWN dedicated capital — no competition for capacity against reversion_bounce "
         "or any other bucket, unlike the blended-portfolio version where this pattern "
         "mostly lost capacity fights under the portfolio's depth-based ordering. Criteria: "
         "TrendState == uptrend, KnifeRiskTier == stabilising, NOT already "
         "trend_continuation, price <= 50-EMA, depth >= -20% vs EMA200, "
         f"days_since_pullback_low >= {EMA_BAND_STABILIZATION_MIN_DAYS}. Same trade "
         "management as every other isolated test here (swing-low/EMA-anchored stop, "
         "Fibonacci-extension target, +2R/give-1R trailing exit) and the same engine/costs/"
         "sizing/sector cap as research/current_trend_gate_ab.py.",
         f"- {len(sig)} raw signals, {sig['ticker'].nunique()} tickers, {START} .. present",
         "- engine: research/current_trend_gate_ab.py's run()/stat() — same costs/sizing/"
         "sector cap as every other isolated test in this repo", "",
         "## Result", "",
         "| | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
         "|---|---|---|---|---|---|---|---|---|"]

    for wn, lo, hi in windows:
        s = stat(res, spy, lo, hi)
        if s is None:
            L.append(f"| {wn} | (insufficient) | | | | | | | |")
            continue
        L.append(f"| {wn} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                  f"{s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
        L.append(f"| {wn} SPY | {s['spy_ret']*100:+.0f}% | | | | | | | |")

    tr = res["tr"]
    if not tr.empty:
        L += ["", "## Exit-reason breakdown", "",
              "| reason | count | avgR |", "|---|---|---|"]
        for reason, g in tr.groupby("reason"):
            L.append(f"| {reason} | {len(g)} | {g['r'].mean():+.2f} |")

    L += ["", "_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship "
          "bias not corrected. This is a first isolated backtest of a brand-new bucket — "
          "treat with the same skepticism as any first result, not as validated the way "
          "trend_continuation/reversion_bounce are._"]

    from pathlib import Path
    out = Path(OUT_PATH)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ema_isolated] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
