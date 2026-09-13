"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Follow-up to research/pullback_shape_ab.py's width-in-bars finding (long/grinding pullbacks
consistently underperformed short/fast ones across every window) — that test only split
width into 3 equal-size terciles, which is enough to say "long is worse" but not enough to
say WHERE the effect actually kicks in or whether it's a smooth gradient or a cliff at a
specific bar count. This sweeps finer quintile buckets (shape of the relationship) and a set
of specific absolute-threshold portfolio variants (same style as
research/current_trend_gate_ab.py's slope sweep) so the Decision Agent's prompt guidance can
cite a specific number instead of a vague tercile edge.

Reuses research.pullback_shape_ab's width_bars computation (same swing-high detection as
core.trend_context.measure_swing_fib_retracement) and caches the tagged signal set so repeat
runs/sweeps don't recompute it from scratch.

Usage:  python -m research.pullback_width_sweep            (uses cache if present)
        python -m research.pullback_width_sweep --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat
from research.pullback_shape_ab import tag_signals

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "pullback_width_signals.pkl"
OUT = Path(__file__).resolve().parent / "pullback_width_sweep.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

# None = no filter (baseline, today's live gate). Others require width_bars <= threshold.
WIDTH_VARIANTS = {
    "no gate (current)": None,
    "width<=10": 10,
    "width<=15": 15,
    "width<=20": 20,
    "width<=25": 25,
    "width<=30": 30,
}


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine/costs/conventions as research/current_trend_gate_ab.py's run(), plus
    carrying width_bars through to closed trades. Same mark-to-market forward-fill fix as
    every other script in this repo."""
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
                               "sector": p["sector"], "width_bars": p["width_bars"]})
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
                                "width_bars": s["width_bars"], "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def quintile_stats(tr: pd.DataFrame) -> pd.DataFrame:
    q = pd.qcut(tr["width_bars"], 5, duplicates="drop")
    rows = []
    for bucket, g in tr.groupby(q, observed=True):
        r = g["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        rows.append({"width_bars_range": str(bucket), "trades": len(g),
                     "win%": (r > 0).mean() * 100, "avgR": r.mean(), "pf": pf})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if SIG_CACHE.exists() and not a.rebuild:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[width_sweep] loaded {len(sig)} cached tagged signals", file=sys.stderr)
    else:
        if not CURRENT_TREND_SIG_CACHE.exists():
            sys.exit("research/data/current_trend_signals.pkl not found — run "
                      "`python -m research.current_trend_gate_ab` first.")
        base = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        print(f"[width_sweep] tagging {len(base)} signals with width_bars...", file=sys.stderr)
        sig = tag_signals(base)
        sig = sig.dropna(subset=["width_bars"]).copy()
        sig["width_bars"] = sig["width_bars"].astype(int)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[width_sweep] tagged + cached {len(sig)} signals", file=sys.stderr)

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    baseline = run(sig, bars, spy_above, calendar, settings)
    qstats = quintile_stats(baseline["tr"])

    L = ["# Pullback width-in-bars — finer sweep (RESEARCH)", "",
         "Follow-up to research/pullback_shape_ab.py's tercile finding (long/grinding "
         "pullbacks underperformed in every window). This sweeps quintiles (finer shape of "
         "the relationship) and specific absolute width<=N thresholds (same style as "
         "research/current_trend_gate_ab.py's slope sweep) to find where the effect actually "
         "kicks in, so Decision Agent prompt guidance can cite a specific bar count.",
         f"- signal set: today's live gate held exactly as-is, {len(sig)} signals with a "
         "computable width_bars", "",
         "## Quintile shape (one portfolio run, today's actual gate, no additional filter)", "",
         "| width_bars range | trades | win% | avgR | PF |",
         "|---|---|---|---|---|"]
    for _, row in qstats.iterrows():
        L.append(f"| {row['width_bars_range']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['pf']:.2f} |")
    L.append("")

    counts = {name: (len(sig) if thr is None else int((sig.width_bars <= thr).sum()))
              for name, thr in WIDTH_VARIANTS.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")
    L.append("## Portfolio-level threshold sweep (require width_bars <= N)")
    L.append("")

    res = {
        name: (baseline if thr is None else run(sig[sig.width_bars <= thr], bars, spy_above, calendar, settings))
        for name, thr in WIDTH_VARIANTS.items()
    }

    for wn, lo, hi in windows:
        L += [f"### {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in WIDTH_VARIANTS:
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

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[width_sweep] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
