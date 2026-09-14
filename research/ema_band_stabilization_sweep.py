"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Follow-up to the EMA-band-pullback validation (research/ema_band_pullback_isolated_ab.py,
validated at days_since_pullback_low >= 10). Raised directly (2026-09-13): is 10 days the
right cutoff, or would a shorter one (catching candidates like UAL/KEY/IRM sooner, which sat
at 6-7 days in a live scan) work almost as well without meaningfully hurting the edge? The
first split (research/ema_band_pullback_v2_ab.py) only tested >=10 vs <10 as one lump
"fresh" bucket — exactly the kind of coarse split that the width-in-bars sweep earlier this
session showed can mislead (a clean-looking 2-bucket split hid a non-monotonic true shape).

Sweeps specific day-count thresholds (>=5, >=7, >=8, >=10, >=12, >=15) as ISOLATED backtests
(own dedicated capital each, no competition for capacity — same methodology as
research/ema_band_pullback_isolated_ab.py, which is what made that result trustworthy) on the
same EMA-band zone (uptrend, stabilising, price <= 50-EMA, depth >= -20% vs EMA200, NOT
already trend_continuation).

Reuses research/ema_band_pullback_v2_ab.py's already-cached, already-tagged signal set — no
new tagging pass needed.

Usage:  python -m research.ema_band_stabilization_sweep
"""

from __future__ import annotations

import sys

import pandas as pd

from config.settings import load_settings
from core.pullback_reversal import PRICE_VS_EMA200_MIN_PCT
from research.current_trend_gate_ab import BARS_DIR, run, stat
from research.ema_band_pullback_v2_ab import SIG_CACHE

OUT_PATH = "research/ema_band_stabilization_sweep.md"
START = "2021-06-01"

# None = no minimum (every EMA-band-zone signal, regardless of duration).
DAY_THRESHOLDS = {
    "no minimum": None,
    ">=5 days": 5,
    ">=7 days": 7,
    ">=8 days": 8,
    ">=10 days (current)": 10,
    ">=12 days": 12,
    ">=15 days": 15,
}


def in_ema_band_zone(row) -> bool:
    """The zone itself, independent of duration — uptrend + stabilising + price <= 50-EMA +
    depth >= -20%, and NOT already trend_continuation (mutually exclusive, same priority as
    classify_setup_type)."""
    if row["tier"] != "stabilising" or row["trend_state"] != "uptrend":
        return False
    if row["in_fib_zone"]:
        return False  # already trend_continuation — excluded from this zone
    return row["price_above_ema50"] is False and row["depth"] >= PRICE_VS_EMA200_MIN_PCT


def main():
    settings = load_settings()
    if not SIG_CACHE.exists():
        sys.exit("research/data/ema_band_v2_signals.pkl not found — run "
                  "`python -m research.ema_band_pullback_v2_ab` first.")
    sig = pd.read_pickle(SIG_CACHE)
    zone = sig[sig.apply(in_ema_band_zone, axis=1)].reset_index(drop=True)
    print(f"[sweep] {len(zone)} raw signals in the EMA-band zone (any duration), "
          f"{zone['ticker'].nunique()} tickers", file=sys.stderr)

    tickers = sorted(zone.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    counts = {name: (len(zone) if thr is None else int((zone.days_since_pullback_low >= thr).sum()))
              for name, thr in DAY_THRESHOLDS.items()}

    res = {
        name: run(zone if thr is None else zone[zone.days_since_pullback_low >= thr],
                   bars, spy_above, calendar, settings)
        for name, thr in DAY_THRESHOLDS.items()
    }

    L = ["# EMA-band pullback — stabilization-duration threshold sweep (RESEARCH)", "",
         "Follow-up to research/ema_band_pullback_isolated_ab.py (validated at "
         "days_since_pullback_low >= 10). Sweeps specific day-count thresholds, each as an "
         "isolated backtest with its own dedicated capital (no competition for capacity) — "
         "checking whether a shorter minimum (catching candidates sooner) holds up nearly as "
         "well, or whether 10 days is doing real, load-bearing work.",
         f"- {len(zone)} raw signals in the EMA-band zone (any duration), "
         f"{zone['ticker'].nunique()} tickers", "",
         "- signal counts per threshold: " + ", ".join(f"{k}={v}" for k, v in counts.items()),
         "", "## Full/train/test/2022 by threshold", ""]

    for wn, lo, hi in windows:
        L += [f"### {wn}", "",
              "| threshold | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in DAY_THRESHOLDS:
            s = stat(res[name], spy, lo, hi)
            if s is None:
                L.append(f"| {name} | (insufficient) | | | | | | | |")
                continue
            sp0 = s["spy_ret"]
            L.append(f"| {name} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                     f"{s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
        if sp0 is not None:
            L.append(f"| SPY | {sp0*100:+.0f}% | | | | | | | |")
        L.append("")

    L += ["_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not "
          "corrected._"]

    from pathlib import Path
    out = Path(OUT_PATH)
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[sweep] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
