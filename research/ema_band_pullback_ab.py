"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Tests an alternative to core.trend_context.classify_setup_type's Fib-zone requirement for
"trend_continuation," raised directly (2026-09-13) after finding a live run where 9 tickers
(IRM, UAL, COLB, UNFI, SEZL, TTC, KEY, FNB, MIRM) were all confirmed uptrend + stabilising but
fell through to SetupType=null because none was in the 38.2-61.8% Fibonacci retracement zone
of its own recent 60-session swing. Checked why: all 9 sit BELOW their 50-day EMA (-1.5% to
-10.1%) but still ABOVE their 200-day EMA (+0.7% to +2.7%) — a genuinely shallow, healthy-
looking dip in a confirmed uptrend. They failed the Fib check because they'd already retraced
63-84% of a SMALL recent 60-day swing, a different (and here, misleading) reference frame from
the broader EMA200 trend structure the "uptrend" read itself is based on.

Alternative zone tested: price at/below the 50-day EMA, down to no worse than the EXISTING
G2 floor already used live (PRICE_VS_EMA200_MIN_PCT, -20%) — i.e. reusing the live gate's own
already-validated depth floor rather than inventing a new one, with the 50-EMA as the new
upper bound in place of the Fib-zone requirement. Classified as a new bucket,
"uptrend_pullback_ema_band," mutually exclusive with the existing trend_continuation
(Fib-zone) definition — checked SEPARATELY, priority trend_continuation > this new bucket >
reversion_bounce > null, so the comparison isolates candidates that ONLY qualify under the new
definition, not ones that already qualified the old way.

Uses today's live-gate signal set (research/data/current_trend_signals.pkl, unmodified) — one
portfolio run, DEFAULT trade management for every bucket (no reversion_bounce-style half-size/
fixed-target applied here — this is about whether the bucket has an edge at all, not about
tuning its trade management, which would be a follow-up once/if this bucket is validated).

Usage:  python -m research.ema_band_pullback_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from core.trend_context import compute_trend_state, measure_swing_fib_retracement
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "ema_band_signals.pkl"
OUT = Path(__file__).resolve().parent / "ema_band_pullback_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

# Reuses the live gate's own already-validated G2 floor — not a new threshold.
EMA_BAND_MIN_DEPTH_PCT = -20.0


def tag_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Adds TrendState/InFibZone/PriceAboveEMA50/Bucket to each cached signal, as of that
    signal's own date (bars up to and including it only — no lookahead)."""
    out_rows = []
    for t, g in sig.groupby("ticker"):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        df_full = compute_indicators(raw.copy()).set_index("Date")
        for _, row in g.iterrows():
            prefix = df_full.loc[:row["date"]]
            trend = compute_trend_state(prefix)
            fib = measure_swing_fib_retracement(prefix)
            out_rows.append({
                **row.to_dict(),
                "trend_state": trend.get("trend_state"),
                "in_fib_zone": fib.get("in_fib_zone"),
                "price_above_ema50": trend.get("price_above_ema50"),
            })
    return pd.DataFrame(out_rows)


def classify(row) -> str | None:
    is_uptrend = row["trend_state"] == "uptrend"
    is_stable = row["tier"] == "stabilising"
    if not is_stable:
        return None
    if is_uptrend and row["in_fib_zone"]:
        return "trend_continuation"
    if is_uptrend and row["price_above_ema50"] is False and row["depth"] >= EMA_BAND_MIN_DEPTH_PCT:
        return "uptrend_pullback_ema_band"
    if row["trend_state"] in ("downtrend", "transitional"):
        return "reversion_bounce"
    return None


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine/costs/conventions as research/current_trend_gate_ab.py's run(), plus
    carrying bucket through to closed trades. Same mark-to-market forward-fill fix as every
    other script in this repo. DEFAULT trade management for every bucket — no
    reversion_bounce-style overrides here."""
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
        rows.append({"bucket": bucket, "trades": len(g), "win%": (r > 0).mean() * 100,
                     "avgR": r.mean(), "medianR": r.median(), "pf": pf})
    return pd.DataFrame(rows)


def main():
    settings = load_settings()
    if SIG_CACHE.exists():
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[ema_band] loaded {len(sig)} cached tagged signals", file=sys.stderr)
    else:
        if not CURRENT_TREND_SIG_CACHE.exists():
            sys.exit("research/data/current_trend_signals.pkl not found — run "
                      "`python -m research.current_trend_gate_ab` first.")
        base = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        print(f"[ema_band] tagging {len(base)} signals...", file=sys.stderr)
        sig = tag_signals(base)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[ema_band] tagged + cached {len(sig)} signals", file=sys.stderr)

    sig["bucket"] = sig.apply(classify, axis=1)
    print(f"[ema_band] bucket counts: {sig['bucket'].value_counts(dropna=False).to_dict()}",
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

    L = ["# EMA-band pullback — an alternative to the Fib-zone requirement for "
         "trend_continuation (RESEARCH)", "",
         "Motivated by a live case: IRM, UAL, COLB, UNFI, SEZL, TTC, KEY, FNB, MIRM were all "
         "confirmed uptrend + stabilising but fell through to SetupType=null because none was "
         "in the 38.2-61.8% Fib-retracement zone of its own recent 60-session swing — all 9 "
         "sit below their 50-EMA (-1.5% to -10.1%) but still above their 200-EMA (+0.7% to "
         "+2.7%), a shallow dip the Fib-zone math misread as 'too deep' relative to a small "
         "recent swing. Tests price <= 50-EMA (down to the existing, already-live G2 floor of "
         "-20% vs EMA200) as an alternative zone, in a NEW bucket "
         "(`uptrend_pullback_ema_band`) checked separately from the existing Fib-zone-based "
         "`trend_continuation` (priority: trend_continuation > this new bucket > "
         "reversion_bounce > null, so this isolates candidates that ONLY qualify the new way).",
         f"- signal set: today's live gate held exactly as-is, {len(sig)} signals",
         f"- bucket counts: {sig['bucket'].value_counts(dropna=False).to_dict()}",
         "- trade management: DEFAULT for every bucket here (no reversion_bounce-style "
         "sizing/exit overrides) — this tests whether the bucket has an edge at all, not how "
         "to size it", "",
         "## Per-bucket trade stats (one portfolio run, default trade management)", "",
         "| bucket | trades | win% | avgR | medianR | PF |",
         "|---|---|---|---|---|---|"]
    for _, row in buckets.iterrows():
        L.append(f"| {row['bucket']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} |")
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
          "corrected. This is a first backtest of a NEW bucket — treat with the same "
          "skepticism as any first result, not as validated the way trend_continuation/"
          "reversion_bounce are (those went through train/test + isolated-portfolio checks "
          "before being trusted)._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ema_band] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
