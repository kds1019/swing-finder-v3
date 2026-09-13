"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Tests a hypothesis raised directly (2026-09-13): a stock can pass every individual-ticker
gate (G1-G4, stabilization) while the SECTOR it belongs to is itself rolling over — and the
claim is that such a pick is less likely to produce a strong return, because the stock is
swimming against its own group's current. This is the well-known "trade with the group, not
against it" principle (Wall Street's version of relative strength / group rotation), but it
had never been checked against this system's own signal set and trade outcomes.

Sector trend is read the SAME way core.trend_context.compute_trend_state() reads an
individual stock's trend (EMA50/EMA200, 20-session EMA200 slope, +-0.5% flat band ->
uptrend/downtrend/transitional) — just applied to the SPDR sector ETF that maps to each
FMP/GICS sector label already attached to every signal, instead of to the stock itself.

Uses the existing live-gate signal set (research/data/current_trend_signals.pkl, built by
research/current_trend_gate_ab.py — G1-G4 + current-trend slope gate, weak-RR dropped; i.e.
EXACTLY what the live screener trades today) rather than rebuilding signals, since this test
is about an ADDITIONAL read (the sector's own trend), not a different entry pattern.

Two views on the same tagged signal set:
  1. Per-bucket raw trade stats (win%/avgR/PF) by sector_trend_state AT ENTRY, from ONE
     portfolio run that takes every live-gate signal (today's actual behavior) — this is the
     direct test of the hypothesis: do trades entered while their sector itself was in a
     downtrend actually do worse than ones entered while the sector was in an uptrend?
  2. Portfolio-level gate variants (no gate / exclude sector-downtrend / sector-uptrend-only)
     across the same full/train/test/2022 windows used everywhere else in this repo — does
     ADDING a sector-trend filter on top of the existing gate actually help the blended
     portfolio, the way the current-trend slope gate did?

Usage:  python -m research.sector_trend_ab            (uses cached signals/bars/ETF history)
        python -m research.sector_trend_ab --refresh-etfs   (re-fetch sector ETF bars)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient

from config.settings import load_settings
from core.indicators import compute_indicators
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from core.trend_context import EMA200_SLOPE_FLAT_BAND_PCT, EMA200_SLOPE_LOOKBACK_DAYS
from research.build_calibration_dataset import get_history
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
OUT = Path(__file__).resolve().parent / "sector_trend_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
ETF_HISTORY_START = datetime(2019, 6, 1, tzinfo=timezone.utc)
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

# One SPDR Select Sector ETF per FMP/GICS sector label already attached to every signal
# (core.universe._FIELD_RENAME's "Sector" column) — an established, standard 1:1 mapping.
SECTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

SECTOR_GATE_VARIANTS = {
    "no gate (current)": None,
    "exclude sector_downtrend": "exclude_downtrend",
    "sector_uptrend_only": "uptrend_only",
}


def _sector_trend_state_series(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version of core.trend_context.compute_trend_state()'s classification,
    applied across an ETF's whole history instead of just its last bar — same thresholds
    (20-session EMA200 slope, +-0.5% flat band)."""
    close = df["Close"].astype(float)
    ema200 = df["EMA200"].astype(float)
    slope = (ema200 / ema200.shift(EMA200_SLOPE_LOOKBACK_DAYS) - 1.0) * 100.0
    above = close > ema200
    state = np.where(
        above & (slope > EMA200_SLOPE_FLAT_BAND_PCT), "uptrend",
        np.where((~above) & (slope < -EMA200_SLOPE_FLAT_BAND_PCT), "downtrend", "transitional"),
    )
    valid = ema200.notna() & slope.notna()
    return pd.DataFrame({"date": df["Date"].values, "state": np.where(valid, state, None)})


def build_sector_trend_table(settings, refresh: bool) -> pd.DataFrame:
    """One row per (date, sector) with that sector ETF's trend_state that day. Long format
    so it can merge_asof against the signal set by sector."""
    client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    rows = []
    for sector, etf in SECTOR_ETF.items():
        raw = get_history(client, etf, ETF_HISTORY_START, refresh=refresh)
        if raw is None or raw.empty:
            print(f"[sector_trend] WARNING: no history for {etf} ({sector}) — skipped", file=sys.stderr)
            continue
        df = compute_indicators(raw.copy())
        st = _sector_trend_state_series(df)
        st["sector"] = sector
        rows.append(st)
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out


def tag_signals_with_sector_trend(sig: pd.DataFrame, sector_trend: pd.DataFrame) -> pd.DataFrame:
    sig = sig.sort_values("date").reset_index(drop=True)
    st = sector_trend.sort_values("date").reset_index(drop=True)
    tagged = pd.merge_asof(sig, st, on="date", by="sector", direction="backward")
    return tagged.rename(columns={"state": "sector_trend_state"})


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine as research/current_trend_gate_ab.py's run(), plus carrying
    sector_trend_state from signal to open position to closed trade so results can be
    bucketed by it afterward. Mark-to-market forward-fills each held position's last known
    close on a day its own bar is missing (see current_trend_gate_ab.py's run() docstring for
    why — same confirmed cached-data gap, same fix)."""
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
                               "sector": p["sector"], "sector_trend_state": p["sector_trend_state"]})
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
                                "sector_trend_state": s["sector_trend_state"], "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def bucket_stats(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, g in tr.groupby("sector_trend_state", dropna=False):
        r = g["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        rows.append({"sector_trend_state": state, "trades": len(g),
                     "win%": (r > 0).mean() * 100, "avgR": r.mean(),
                     "medianR": r.median(), "pf": pf})
    return pd.DataFrame(rows).sort_values("sector_trend_state", key=lambda s: s.fillna("zzz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-etfs", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if not CURRENT_TREND_SIG_CACHE.exists():
        sys.exit("research/data/current_trend_signals.pkl not found — run "
                  "`python -m research.current_trend_gate_ab` first to build it.")
    sig = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    print(f"[sector_trend] {len(sig)} live-gate signals loaded", file=sys.stderr)

    sector_trend = build_sector_trend_table(settings, refresh=a.refresh_etfs)
    print(f"[sector_trend] sector ETF trend table: {len(sector_trend)} (date, sector) rows",
          file=sys.stderr)

    sig = tag_signals_with_sector_trend(sig, sector_trend)
    unmatched = sig["sector_trend_state"].isna().sum()
    print(f"[sector_trend] {unmatched}/{len(sig)} signals had no sector-trend match "
          f"(ETF history too short that far back) — dropped from bucketed view, "
          f"kept as 'no gate' in the portfolio view", file=sys.stderr)

    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    # --- View 1: one portfolio run on today's actual live-gate signal set, bucketed by
    # the sector's trend_state at entry (the direct hypothesis test) ---
    baseline = run(sig, bars, spy_above, calendar, settings)
    buckets = bucket_stats(baseline["tr"])

    L = ["# Sector-trend read — does the STOCK's own SECTOR trending down predict a weaker "
         "return? (RESEARCH)", "",
         "Hypothesis (raised directly 2026-09-13): a pick that clears every individual-stock "
         "gate while its own SECTOR is rolling over is less likely to produce a strong return "
         "— trading against the group, not with it. Sector trend read the same way "
         "core.trend_context.compute_trend_state() reads an individual stock (EMA50/EMA200, "
         "20-session EMA200 slope, +-0.5% flat band), applied to the SPDR sector ETF matching "
         "each signal's FMP/GICS sector label instead of the stock itself.",
         f"- signal set: today's live gate (research/data/current_trend_signals.pkl) held "
         f"exactly as-is, {len(sig)} signals, {unmatched} unmatched (dropped from the bucket "
         "view below)",
         "- engine: research/current_trend_gate_ab.py's run()/stat() conventions", "",
         "## View 1 — realized trade outcomes, bucketed by sector_trend_state at entry "
         "(one portfolio run, today's actual gate, no additional filter)", "",
         "| sector_trend_state | trades | win% | avgR | medianR | PF |",
         "|---|---|---|---|---|---|"]
    for _, row in buckets.iterrows():
        L.append(f"| {row['sector_trend_state']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} |")
    L.append("")

    # --- View 2: portfolio-level gate variants across the standard windows ---
    sig_variants = {
        "no gate (current)": sig,
        "exclude sector_downtrend": sig[sig.sector_trend_state != "downtrend"],
        "sector_uptrend_only": sig[sig.sector_trend_state == "uptrend"],
    }
    res = {name: (baseline if name == "no gate (current)"
                  else run(s, bars, spy_above, calendar, settings))
           for name, s in sig_variants.items()}

    counts = {name: len(s) for name, s in sig_variants.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")
    L.append("## View 2 — portfolio-level gate variants (does FILTERING on sector trend help "
             "the blended portfolio?)")
    L.append("")

    for wn, lo, hi in windows:
        L += [f"### {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in SECTOR_GATE_VARIANTS:
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
          "corrected. Sector ETF history starts 2019-06-01 so the 220-bar trend-state minimum "
          "is satisfied well before this backtest's 2021-06-01 signal start._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[sector_trend] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
