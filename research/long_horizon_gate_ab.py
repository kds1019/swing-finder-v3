"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated portfolio test of a SECOND new candidate gate, layered ON TOP of the already-
tested current-trend gate (research/current_trend_gate_ab.py, slope20 >= -2%, which won in
every window there and is the leading candidate for promotion). This one targets a
DIFFERENT failure mode that slope20 >= -2% does NOT catch.

Motivated by a live case (ENPH, 2026-09-11): the 126-day G1 gate read a strong uptrend
(+7.5%), and the current 20-day slope was -1.99% — just inside the -2% floor, so it still
passes that gate. But tracing the actual EMA200 shape over the full year showed a decline
into a low ~157 sessions ago, a sharp recovery, and a stall right at the top of that
recovery — a V-shaped round trip, not a sustained trend. The 126-day window happened to
anchor its "before" point during the recovery leg (after the low), so it reads as a clean
uptrend even though the LONGER 252-day (1-year) slope is only +2.4% — much weaker. That
gap (short window strong, long window weak) is the signature of a stall-at-the-top-of-a-
bounce, which neither the existing G1 gate nor the already-tested current-trend gate catches.

Tests whether ALSO requiring the 252-day EMA200 slope not be too far below the 126-day one
(or below its own floor) helps, hurts, or washes — layered on top of slope20>=-2%, not
instead of it.

Same engine/costs/conventions as research/current_trend_gate_ab.py — reuses that script's
cached signal set (research/data/current_trend_signals.pkl) and adds slope252 per signal by
re-reading each ticker's cached bars once (fast — no re-running the expensive gate match).

Splits: full, 2021-2024 (train), 2025-2026 (test), 2022 alone.

Usage:  python -m research.long_horizon_gate_ab            (uses current_trend_signals.pkl cache)
        python -m research.long_horizon_gate_ab --rebuild-slope252   (recompute slope252 only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_CACHE, TIER_ORDER, run, stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "long_horizon_signals.pkl"
OUT = Path(__file__).resolve().parent / "long_horizon_gate_ab.md"

INITIAL = 100_000.0
START = "2021-06-01"
SLOPE20_FLOOR = -2.0  # the already-tested, leading-candidate gate — held fixed as the baseline here
SLOPE252_LOOKBACK_DAYS = 252

# None = no additional filter beyond slope20>=-2% (this test's own baseline). Others also
# require the 252-session EMA200 slope to clear the given floor.
SLOPE252_VARIANTS = {
    "slope20>=-2% only (leading candidate)": None,
    "+ slope252>=0%": 0.0,
    "+ slope252>=1%": 1.0,
    "+ slope252>=2%": 2.0,
    "+ slope252>=3%": 3.0,
}


def add_slope252(sig: pd.DataFrame) -> pd.DataFrame:
    """Adds a slope252 column to the cached current_trend_gate_ab signal set by re-reading
    each ticker's cached bars once and computing the 252-session EMA200 slope at each
    signal's date — cheap relative to rebuilding the whole signal set from scratch."""
    sig = sig.copy()
    sig["slope252"] = np.nan
    for t in sig["ticker"].unique():
        cache = BARS_DIR / f"{t}.pkl"
        if not cache.exists():
            continue
        raw = pd.read_pickle(cache)
        df = compute_indicators(raw.copy())
        df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        ema200 = df["EMA200"]
        slope252 = (ema200 / ema200.shift(SLOPE252_LOOKBACK_DAYS) - 1.0) * 100.0
        lookup = dict(zip(df["Date"], slope252))
        mask = sig["ticker"] == t
        sig.loc[mask, "slope252"] = sig.loc[mask, "date"].map(lookup)
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-slope252", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if not CURRENT_TREND_CACHE.exists():
        raise SystemExit(f"Run research.current_trend_gate_ab first to build {CURRENT_TREND_CACHE}")

    if SIG_CACHE.exists() and not a.rebuild_slope252:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[longhorizon] loaded {len(sig)} cached signals (with slope252)", file=sys.stderr)
    else:
        base = pd.read_pickle(CURRENT_TREND_CACHE)
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        sig = add_slope252(base)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[longhorizon] built + cached {len(sig)} signals with slope252", file=sys.stderr)

    # This test's own population: everything that already clears the leading-candidate
    # slope20>=-2% gate (that gate is being treated as already-adopted here; this test asks
    # whether adding slope252 on top of it helps further, not whether slope20 itself helps).
    sig = sig[sig["slope20"] >= SLOPE20_FLOOR].reset_index(drop=True)

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    L = ["# Long-horizon gate (252-session EMA200 slope) — isolated portfolio A/B (RESEARCH)", "",
         "Layered ON TOP of the already-tested slope20>=-2% gate (research/current_trend_gate_ab.py), "
         "not instead of it. Motivated by ENPH (2026-09-11): 126-day slope +7.5% (\"uptrend\"), current "
         "20-day slope -1.99% (just inside the -2% floor, still passes), but 252-day slope only +2.4% "
         "— a V-shaped recovery stalling at its own recent high, which neither existing check catches.",
         f"- base population: {len(sig)} signals that already clear slope20>=-2%.",
         "- engine: same as current_trend_gate_ab.py / g1_ab.py; pool ordered knife-tier then depth", ""]

    counts = {name: int((sig.slope252 >= thr).sum()) if thr is not None else len(sig)
              for name, thr in SLOPE252_VARIANTS.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")

    res = {
        name: run(sig if thr is None else sig[sig.slope252 >= thr], bars, spy_above, calendar, settings)
        for name, thr in SLOPE252_VARIANTS.items()
    }

    for wn, lo, hi in windows:
        L += [f"## {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in SLOPE252_VARIANTS:
            s = stat(res[name], spy, lo, hi)
            if s is None:
                L.append(f"| {name} | (insufficient) | | | | | | | |"); continue
            sp0 = s["spy_ret"]
            L.append(f"| {name} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                     f"{s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
        if sp0 is not None:
            L.append(f"| SPY | {sp0*100:+.0f}% | | | | | | | |")
        L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[longhorizon] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
