"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Follow-up to research/ema_band_pullback_ab.py, which tested "price below the 50-EMA, no
worse than -20% vs the 200-EMA, uptrend, stabilising" and found it a net loser (PF 0.95) —
NOT because the reward side was bad (winners averaged +1.5R to +3.7R, matching sane trade
plans like UAL's real 3.15 R:R), but because 68% of trades were stopped out directly. Raised
directly (2026-09-13): require the stabilization to have HELD for at least 10 days (not just
the moment KnifeRiskTier first reads "stabilising", which can be day 1-2 of a bounce and
prone to reversing straight through the stop) before treating it as a real base, using
core.pullback_reversal.measure_stabilization's own `days_since_pullback_low` (bars since the
pullback's low point) — a field the live screener already computes but doesn't gate on.

Same zone as before (price <= 50-EMA, depth >= -20% vs EMA200, TrendState == uptrend,
KnifeRiskTier == stabilising), now split into two sub-buckets to isolate the effect of the
duration filter specifically:
  - uptrend_pullback_ema_band_10d : days_since_pullback_low >= 10
  - uptrend_pullback_ema_band_fresh : the same zone, days_since_pullback_low < 10 (i.e. what
    the original test's bucket looked like BEFORE this split — should reproduce that bucket's
    poor result if the duration filter is what matters, confirming rather than just asserting)

Uses today's live-gate signal set (research/data/current_trend_signals.pkl, unmodified), one
portfolio run, DEFAULT trade management for every bucket (same as the first EMA-band test —
this is about whether the bucket has an edge at all before any sizing/exit tuning).

Usage:  python -m research.ema_band_pullback_v2_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import measure_stabilization
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from core.trend_context import compute_trend_state, measure_swing_fib_retracement
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "ema_band_v2_signals.pkl"
OUT = Path(__file__).resolve().parent / "ema_band_pullback_v2_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

EMA_BAND_MIN_DEPTH_PCT = -20.0
MIN_STABILIZATION_DAYS = 10


def tag_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Adds TrendState/InFibZone/PriceAboveEMA50/DaysSincePullbackLow to each cached signal,
    as of that signal's own date only (no lookahead)."""
    out_rows = []
    for t, g in sig.groupby("ticker"):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        df_full = compute_indicators(raw.copy())
        df_indexed = df_full.set_index("Date")
        for _, row in g.iterrows():
            prefix = df_indexed.loc[:row["date"]]
            trend = compute_trend_state(prefix)
            fib = measure_swing_fib_retracement(prefix)
            stab = measure_stabilization(prefix.reset_index())
            out_rows.append({
                **row.to_dict(),
                "trend_state": trend.get("trend_state"),
                "in_fib_zone": fib.get("in_fib_zone"),
                "price_above_ema50": trend.get("price_above_ema50"),
                "days_since_pullback_low": stab.get("days_since_pullback_low"),
            })
    return pd.DataFrame(out_rows)


def classify(row) -> str | None:
    is_uptrend = row["trend_state"] == "uptrend"
    is_stable = row["tier"] == "stabilising"
    if not is_stable:
        return None
    if is_uptrend and row["in_fib_zone"]:
        return "trend_continuation"
    in_ema_band = (is_uptrend and row["price_above_ema50"] is False
                   and row["depth"] >= EMA_BAND_MIN_DEPTH_PCT)
    if in_ema_band:
        dspl = row["days_since_pullback_low"]
        if dspl is not None and dspl >= MIN_STABILIZATION_DAYS:
            return "uptrend_pullback_ema_band_10d"
        return "uptrend_pullback_ema_band_fresh"
    if row["trend_state"] in ("downtrend", "transitional"):
        return "reversion_bounce"
    return None


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine/costs/conventions as research/current_trend_gate_ab.py's run(), plus
    carrying bucket through to closed trades. Same mark-to-market forward-fill fix as every
    other script in this repo. DEFAULT trade management for every bucket."""
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
                               "sector": p["sector"], "bucket": p["bucket"]})
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
                                "bucket": s["bucket"], "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def bucket_stats(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, g in tr.groupby("bucket", dropna=False):
        r = g["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        exit_reasons = g["reason"].value_counts().to_dict()
        rows.append({"bucket": bucket, "trades": len(g), "win%": (r > 0).mean() * 100,
                     "avgR": r.mean(), "medianR": r.median(), "pf": pf,
                     "stop_hit_pct": exit_reasons.get("stop_hit", 0) / len(g) * 100})
    return pd.DataFrame(rows)


def main():
    settings = load_settings()
    if SIG_CACHE.exists():
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[ema_v2] loaded {len(sig)} cached tagged signals", file=sys.stderr)
    else:
        if not CURRENT_TREND_SIG_CACHE.exists():
            sys.exit("research/data/current_trend_signals.pkl not found — run "
                      "`python -m research.current_trend_gate_ab` first.")
        base = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        print(f"[ema_v2] tagging {len(base)} signals...", file=sys.stderr)
        sig = tag_signals(base)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[ema_v2] tagged + cached {len(sig)} signals", file=sys.stderr)

    sig["bucket"] = sig.apply(classify, axis=1)
    print(f"[ema_v2] bucket counts: {sig['bucket'].value_counts(dropna=False).to_dict()}",
          file=sys.stderr)

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    baseline = run(sig, bars, spy_above, calendar, settings)
    buckets = bucket_stats(baseline["tr"])

    L = ["# EMA-band pullback v2 — does requiring 10+ days of stabilization fix the "
         "stop-out rate? (RESEARCH)", "",
         "Follow-up to research/ema_band_pullback_ab.py, which found the EMA-band bucket "
         "(price <= 50-EMA, no worse than -20% vs 200-EMA, uptrend, stabilising) a net loser "
         "(PF 0.95) because 68% of trades were stopped out directly, NOT because the reward "
         "side was bad (winners averaged +1.5R to +3.7R). Tests whether requiring the "
         "stabilization to have HELD for >= 10 days (core.pullback_reversal.measure_"
         "stabilization's days_since_pullback_low, already computed live but not gated on) "
         "fixes this, by splitting the same zone into a >=10-day sub-bucket and a <10-day "
         "('fresh') sub-bucket.",
         f"- signal set: today's live gate held exactly as-is, {len(sig)} signals",
         f"- bucket counts: {sig['bucket'].value_counts(dropna=False).to_dict()}",
         "- trade management: DEFAULT for every bucket (no sizing/exit overrides)", "",
         "## Per-bucket trade stats (one portfolio run, default trade management)", "",
         "| bucket | trades | win% | avgR | medianR | PF | % stopped out |",
         "|---|---|---|---|---|---|---|"]
    for _, row in buckets.iterrows():
        L.append(f"| {row['bucket']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} | "
                 f"{row['stop_hit_pct']:.0f}% |")
    L.append("")

    L.append("## By year and bucket")
    L.append("")
    L.append("| year | bucket | trades | win% | avgR |")
    L.append("|---|---|---|---|---|")
    tr = baseline["tr"]
    if not tr.empty:
        tr["yr"] = pd.to_datetime(tr["exit_date"]).dt.year
        for (y, bucket), g in tr.groupby(["yr", "bucket"], dropna=False):
            L.append(f"| {y} | {bucket} | {len(g)} | {(g.r>0).mean()*100:.0f}% | {g.r.mean():+.2f} |")
    L.append("")

    L += ["_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not "
          "corrected. First backtest of the 10-day split — treat with the same skepticism as "
          "any first result._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ema_v2] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
