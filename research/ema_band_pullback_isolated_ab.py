"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated portfolio test of the "uptrend_pullback_ema_band_10d" bucket from
research/ema_band_pullback_v2_ab.py (price <= 50-EMA, no worse than -20% vs 200-EMA,
TrendState == uptrend, KnifeRiskTier == stabilising, stabilization held >= 10 days per
core.pullback_reversal.measure_stabilization's days_since_pullback_low) on its OWN — no
competition for capacity against reversion_bounce or any other bucket.

Motivated directly: the same bucket in a blended portfolio run only got 11 of its 318 raw
signals actually traded (avg R +0.20, PF 1.31 — encouraging but too small a sample to trust),
because it was competing for the day's limited position slots against reversion_bounce
signals and mostly losing those fights under the portfolio's depth-based ordering. This test
gives every one of the 318 raw signals a fair shot with its own dedicated capital, the same
way research/pullback_to_50_ema_ab.py and every other standalone pattern test in this repo
was run — isolating whether the PATTERN itself has a real edge, separate from how it competes
for capacity inside the blended live portfolio.

Reuses research/current_trend_gate_ab.py's run()/stat() engine (same costs/sizing/sector cap/
regime filter as every other isolated test here) and research/ema_band_pullback_v2_ab.py's
already-cached, already-tagged signal set + classify() — no new tagging pass needed.

Usage:  python -m research.ema_band_pullback_isolated_ab
"""

from __future__ import annotations

import sys

import pandas as pd

from config.settings import load_settings
from research.current_trend_gate_ab import BARS_DIR, run, stat
from research.ema_band_pullback_v2_ab import SIG_CACHE, classify

OUT_PATH = "research/ema_band_pullback_isolated_ab.md"
START = "2021-06-01"
BUCKET = "uptrend_pullback_ema_band_10d"


def main():
    settings = load_settings()
    if not SIG_CACHE.exists():
        sys.exit("research/data/ema_band_v2_signals.pkl not found — run "
                  "`python -m research.ema_band_pullback_v2_ab` first.")
    sig = pd.read_pickle(SIG_CACHE)
    sig["bucket"] = sig.apply(classify, axis=1)
    sig = sig[sig["bucket"] == BUCKET].reset_index(drop=True)
    print(f"[ema_isolated] {len(sig)} raw signals in {BUCKET}, "
          f"{sig['ticker'].nunique()} tickers", file=sys.stderr)

    if sig.empty:
        sys.exit(f"No signals in bucket {BUCKET} — nothing to backtest.")

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    res = run(sig, bars, spy_above, calendar, settings)

    L = ["# EMA-band pullback (10+ day stabilization) — standalone isolated backtest "
         "(RESEARCH)", "",
         "The uptrend_pullback_ema_band_10d bucket from research/ema_band_pullback_v2_ab.py, "
         "tested on its OWN dedicated capital — no competition for capacity against "
         "reversion_bounce or any other bucket, unlike the blended-portfolio version where "
         "only 11 of 318 raw signals ever got traded. Criteria: TrendState == uptrend, "
         "KnifeRiskTier == stabilising, price <= 50-EMA, depth >= -20% vs EMA200, "
         "days_since_pullback_low >= 10. Same trade management as every other isolated test "
         "here (swing-low/EMA-anchored stop, Fibonacci-extension target, +2R/give-1R trailing "
         "exit) and the same engine/costs/sizing/sector cap as "
         "research/current_trend_gate_ab.py.",
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
