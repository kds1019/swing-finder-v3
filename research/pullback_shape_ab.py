"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Tests two pullback-SHAPE features together, per direct request (2026-09-13): does the depth
of a pullback have a nonlinear ("sweet spot") relationship with outcome rather than the
"deeper is better" read implied by using it as a plain sort key, and does the pullback's
WIDTH IN BARS (how long the whole decline-to-stabilization round trip took, not just how many
days it's rested since the low) predict anything on its own. Neither is currently used this
way: `depth` (price_vs_ema200_pct) is only ever used as a continuous tie-break sort key
(deepest first) when signals compete for capacity; nothing in the codebase measures the total
age of a pullback back to its actual peak — core.pullback_reversal.measure_stabilization's
own `days_since_pullback_low` only counts forward from the LOW to today, not back to the peak
that started the decline.

Width-in-bars reuses the exact swing-high detection already used and validated for the
Fibonacci-zone read (core.trend_context.measure_swing_fib_retracement: SWING_LOOKBACK_DAYS=60,
SWING_PIVOT_LEFT_RIGHT_BARS=5, via core.indicators.find_pivot_points) rather than inventing a
new peak-finding method — the most recent confirmed pivot high within the last 60 sessions is
"the peak," and width_bars is simply how many bars ago that was, recomputed as of each
signal's own date (no lookahead — same window a live run would have seen that day).

Uses today's live-gate signal set (research/data/current_trend_signals.pkl, unmodified) —
`depth` is already a column on every signal there; width_bars is computed fresh per signal
here. Both features are split into terciles (by their own realized distribution, reported in
the output) rather than guessed fixed thresholds, since neither's real distribution was known
ahead of time.

Two views, per this repo's usual convention:
  1. Raw bucketed trade outcomes (win%/avgR/PF) by depth-tercile, by width-tercile, and the
     joint depth x width breakdown — from ONE portfolio run of today's actual live gate
     (no additional filter).
  2. Portfolio-level restriction variants (does keeping only one tercile of depth, or of
     width, change the blended portfolio result) across the standard full/train/test/2022
     windows.

Usage:  python -m research.pullback_shape_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import find_pivot_points
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from core.trend_context import SWING_LOOKBACK_DAYS, SWING_PIVOT_LEFT_RIGHT_BARS
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
OUT = Path(__file__).resolve().parent / "pullback_shape_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}


def compute_width_bars(prefix: pd.DataFrame) -> int | None:
    """Bars since the most recent confirmed swing high (same detection as
    core.trend_context.measure_swing_fib_retracement) to the last bar of `prefix` — how long
    ago the pullback's actual peak was, not how long it's rested since the low."""
    if len(prefix) < SWING_LOOKBACK_DAYS:
        return None
    window = prefix.tail(SWING_LOOKBACK_DAYS).reset_index(drop=True)
    pivots = find_pivot_points(window, left_bars=SWING_PIVOT_LEFT_RIGHT_BARS,
                                right_bars=SWING_PIVOT_LEFT_RIGHT_BARS)
    phs = pivots["pivot_highs"]
    swing_high_bar = max(phs, key=lambda p: p["bar"])["bar"] if phs else int(window["High"].to_numpy().argmax())
    return int(len(window) - 1 - swing_high_bar)


def tag_signals(sig: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for t, g in sig.groupby("ticker"):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date")
        for _, row in g.iterrows():
            d = row["date"]
            prefix = raw.loc[:d]
            width = compute_width_bars(prefix)
            out_rows.append({**row.to_dict(), "width_bars": width})
    return pd.DataFrame(out_rows)


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine/costs/conventions as research/current_trend_gate_ab.py's run(), plus
    carrying depth_bucket/width_bucket from signal to open position to closed trade. Same
    mark-to-market forward-fill fix as every other script in this repo (see
    current_trend_gate_ab.py's run() docstring)."""
    d = sig.assign(_tr=sig.tier.map(TIER_ORDER).fillna(2))
    by_date = {dt: g.sort_values(["_tr", "depth"]) for dt, g in d.groupby("date")}
    frict = SLIP_BPS / 10000.0
    cash, positions, eq, closed = INITIAL, {}, [], []
    for dt in calendar:
        for t in list(positions):
            p = positions[t]
            if dt not in bars[t].index:
                continue
            b = bars[t].loc[dt]
            hi, lo, cl = float(b["High"]), float(b["Low"]), float(b["Close"])
            p["last_close"] = cl
            risk = p["entry"] - p["stop"]
            eff = max(p["stop"], p["peak"] - TRAIL_GIVEBACK_R * risk) if p["active"] else p["stop"]
            p["held"] += 1
            xp = xr = None
            if lo <= eff:
                xp, xr = eff, ("trail_stop" if eff > p["stop"] else "stop_hit")
            elif hi >= p["target"]:
                xp, xr = p["target"], "target_hit"
            elif p["held"] >= MAX_HOLD:
                xp, xr = cl, "expired"
            p["peak"] = max(p["peak"], hi)
            if not p["active"] and hi >= p["entry"] + TRAIL_ACTIVATE_R * risk:
                p["active"] = True
            if xp is not None:
                cash += p["shares"] * xp * (1 - frict)
                closed.append({"exit_date": dt, "reason": xr, "r": (xp - p["entry"]) / risk,
                               "sector": p["sector"], "depth_bucket": p["depth_bucket"],
                               "width_bucket": p["width_bucket"]})
                del positions[t]
        mtm = cash + sum(pp["shares"] * pp["last_close"] for pp in positions.values())
        eq.append((dt, mtm))
        if dt in by_date and bool(spy_above.get(dt, False)) and len(positions) < MAX_POS:
            sec = {}
            for pp in positions.values():
                sec[pp["sector"]] = sec.get(pp["sector"], 0) + 1
            for _, s in by_date[dt].iterrows():
                t = s["ticker"]
                if t in positions or len(positions) >= MAX_POS or sec.get(s["sector"], 0) >= SECTOR_CAP:
                    continue
                entry = s["entry"] * (1 + frict)
                rps = entry - s["stop"]
                if rps <= 0:
                    continue
                sh = int(min((mtm * settings.risk_per_trade_pct / 100) / rps,
                             (mtm * MAX_POS_PCT / 100) / entry, cash / entry))
                if sh <= 0:
                    continue
                cash -= sh * entry
                positions[t] = {"shares": sh, "entry": entry, "stop": s["stop"], "target": s["target"],
                                "peak": entry, "active": False, "held": 0, "sector": s["sector"],
                                "depth_bucket": s["depth_bucket"], "width_bucket": s["width_bucket"],
                                "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def bucket_stats(tr: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for state, g in tr.groupby(col, dropna=False, observed=True):
        r = g["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        rows.append({col: state, "trades": len(g), "win%": (r > 0).mean() * 100,
                     "avgR": r.mean(), "medianR": r.median(), "pf": pf})
    return pd.DataFrame(rows)


def main():
    settings = load_settings()
    if not CURRENT_TREND_SIG_CACHE.exists():
        sys.exit("research/data/current_trend_signals.pkl not found — run "
                  "`python -m research.current_trend_gate_ab` first to build it.")
    sig = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    print(f"[pullback_shape] {len(sig)} live-gate signals loaded", file=sys.stderr)

    sig = tag_signals(sig)
    unmatched = sig["width_bars"].isna().sum()
    print(f"[pullback_shape] width_bars computed, {unmatched}/{len(sig)} unmatched "
          f"(insufficient history)", file=sys.stderr)
    sig = sig.dropna(subset=["width_bars"]).copy()
    sig["width_bars"] = sig["width_bars"].astype(int)

    sig["depth_bucket"] = pd.qcut(sig["depth"], 3, labels=["deep", "moderate", "shallow"])
    sig["width_bucket"] = pd.qcut(sig["width_bars"], 3, labels=["short", "medium", "long"])
    depth_edges = pd.qcut(sig["depth"], 3).cat.categories
    width_edges = pd.qcut(sig["width_bars"], 3).cat.categories
    print(f"[pullback_shape] depth terciles: {list(depth_edges)}", file=sys.stderr)
    print(f"[pullback_shape] width terciles (bars): {list(width_edges)}", file=sys.stderr)

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    baseline = run(sig, bars, spy_above, calendar, settings)
    depth_buckets = bucket_stats(baseline["tr"], "depth_bucket")
    width_buckets = bucket_stats(baseline["tr"], "width_bucket")
    joint = baseline["tr"].groupby(["depth_bucket", "width_bucket"], observed=True)["r"].agg(
        trades="count", win_pct=lambda r: (r > 0).mean() * 100, avgR="mean")

    L = ["# Pullback shape — depth regime + width-in-bars (RESEARCH)", "",
         "Tests whether depth (price_vs_ema200_pct) has a nonlinear/sweet-spot relationship "
         "with outcome rather than the 'deeper is better' read implied by using it as a plain "
         "sort key, and whether width-in-bars (bars since the pullback's actual peak, via the "
         "same swing-high detection used for the Fibonacci-zone read — NOT "
         "days_since_pullback_low, which only counts forward from the low) predicts anything "
         "on its own. Neither is currently used this way in the live gate.",
         f"- signal set: today's live gate held exactly as-is, {len(sig)} signals with a "
         f"computable width_bars ({unmatched} dropped for insufficient history)",
         f"- depth terciles: {list(depth_edges)}",
         f"- width terciles (bars since peak): {list(width_edges)}", "",
         "## View 1 — realized trade outcomes (one portfolio run, today's actual gate, no "
         "additional filter)", "",
         "### by depth tercile", "",
         "| depth_bucket | trades | win% | avgR | medianR | PF |",
         "|---|---|---|---|---|---|"]
    for _, row in depth_buckets.iterrows():
        L.append(f"| {row['depth_bucket']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} |")
    L += ["", "### by width tercile", "",
          "| width_bucket | trades | win% | avgR | medianR | PF |",
          "|---|---|---|---|---|---|"]
    for _, row in width_buckets.iterrows():
        L.append(f"| {row['width_bucket']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} |")
    L += ["", "### joint depth x width (trades / win% / avgR)", "",
          "| depth_bucket | width_bucket | trades | win% | avgR |",
          "|---|---|---|---|---|"]
    for (db, wb), row in joint.iterrows():
        L.append(f"| {db} | {wb} | {row['trades']:.0f} | {row['win_pct']:.0f}% | {row['avgR']:+.2f} |")
    L.append("")

    sig_variants = {"no gate (current)": sig}
    for tercile in ["deep", "moderate", "shallow"]:
        sig_variants[f"depth={tercile} only"] = sig[sig.depth_bucket == tercile]
    for tercile in ["short", "medium", "long"]:
        sig_variants[f"width={tercile} only"] = sig[sig.width_bucket == tercile]

    res = {name: (baseline if name == "no gate (current)"
                  else run(s, bars, spy_above, calendar, settings))
           for name, s in sig_variants.items()}
    counts = {name: len(s) for name, s in sig_variants.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")
    L.append("## View 2 — portfolio-level restriction variants (does keeping only one "
             "tercile change the blended portfolio result?)")
    L.append("")

    for wn, lo, hi in windows:
        L += [f"### {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in sig_variants:
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
          "corrected. Terciles are of THIS signal set's own realized distribution, not fixed "
          "a-priori thresholds — see the edges reported above._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[pullback_shape] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
