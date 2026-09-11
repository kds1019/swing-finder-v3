"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated portfolio test of a NEW candidate gate: reject screener matches whose CURRENT
EMA200 slope (20 sessions, core.trend_context.EMA200_SLOPE_LOOKBACK_DAYS) has already
rolled over negative, on top of the existing G1 gate (EMA200 risen >= EMA200_MIN_UPTREND_PCT
over the last 126 sessions).

Motivated by a live finding (2026-09-11): 24 of 26 candidates in a real full-universe scan
had the 126-day G1 gate reading "uptrend" while the CURRENT 20-day EMA200 slope had already
turned negative for that same ticker — the slow 126-day lookback (deliberately chosen so a
temporary flattening during a genuine pullback doesn't reject it — see core/pullback_reversal
.py's module docstring) can't yet distinguish "temporary dip in a real uptrend" from
"actually rolled over into a downtrend." SPY itself was positive across every window that
same day (5d/20d/60d/126d all >0), so this isn't a broad market rollover — it's specific to
what this screener selects. This tests whether requiring the CURRENT slope not be sharply
negative — ON TOP OF, not instead of, the existing gate — helps, hurts, or washes at the
portfolio level. Live track record at the time of this test: 231 decisive picks over
2026-07-09..2026-09-11, 16.5% win rate, -2.13% avg return — a real, non-noise negative
number, not a reason to trust instinct over a backtest here either.

Same engine/costs/conventions as research/g1_ab.py / research/portfolio_backtest.py — G1/G2/
G3/G4 held at CURRENT live values, weak-RR dropped, pool ordered knife-tier then depth. Only
the new current-slope filter is varied.

Signals built once (slope20 kept as a column) and cached to research/data/current_trend_signals.pkl.

Splits: full, 2021-2024 (train), 2025-2026 (test), 2022 alone.

Usage:  python -m research.current_trend_gate_ab            (uses cache if present)
        python -m research.current_trend_gate_ab --rebuild
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
    MIN_BARS_FOR_SCREENER, PRICE_VS_EMA200_MAX_PCT, PRICE_VS_EMA200_MIN_PCT,
    EMA200_MIN_UPTREND_PCT, detect_pullback_reversal, measure_stabilization,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from core.trend_context import EMA200_SLOPE_LOOKBACK_DAYS
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "current_trend_signals.pkl"
OUT = Path(__file__).resolve().parent / "current_trend_gate_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD, WINDOW = 20.0, 6, 3, 5, 30, 300
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

# None = no filter (baseline, matches today's live gate exactly). Others reject any signal
# whose current 20-session EMA200 slope is below the threshold, on top of everything else.
SLOPE_VARIANTS = {
    "no gate (current)": None,
    "slope>=-2%": -2.0,
    "slope>=-0.5% (trend_context flat-band)": -0.5,
    "slope>=0%": 0.0,
    "slope>=2%": 2.0,
}


def build_signals(settings) -> pd.DataFrame:
    """Same gate as the live screener (G1/G2/G3/G4 + weak-RR drop), unrelaxed — this tests
    an ADDITIONAL filter layered on top, not a replacement, so the baseline signal set must
    exactly match what's live today. Tags each signal with slope20 (the current 20-session
    EMA200 slope) so the additional filter can be varied downstream without rebuilding."""
    uni = build_universe(settings)
    sectors = dict(zip(uni["Ticker"], uni["Sector"]))
    tickers = sorted(p.stem for p in BARS_DIR.glob("*.pkl")
                      if p.stem != "SPY" and p.stem in sectors)
    print(f"[curtrend] {len(tickers)} tickers", file=sys.stderr)
    sig = []
    for n, t in enumerate(tickers, 1):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        slope20 = (ema200 / ema200.shift(EMA200_SLOPE_LOOKBACK_DAYS) - 1.0) * 100.0
        net = (upt >= EMA200_MIN_UPTREND_PCT) & pvs.between(PRICE_VS_EMA200_MIN_PCT - 2, PRICE_VS_EMA200_MAX_PCT + 2)
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1:
                continue
            d = df["Date"].iloc[i]
            if str(d.date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            if not detect_pullback_reversal(prefix).get("detected"):
                continue
            try:
                plan = compute_trade_plan(prefix, settings)
            except ZeroDivisionError:
                # core.trade_plan.find_support_resistance.cluster() can divide by a ~0 pivot
                # price on deep-split-adjusted ancient bars — same known issue as g1_ab.py.
                continue
            if plan is None or plan["stop"] >= plan["entry"] or plan["weak_rr"]:
                continue
            stab = measure_stabilization(prefix)
            s20 = float(slope20.iloc[i])
            sig.append({
                "date": d, "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "depth": float(pvs.iloc[i]), "tier": stab.get("knife_risk_tier"),
                "slope20": round(s20, 2) if pd.notna(s20) else None,
            })
        if n % 100 == 0:
            print(f"[curtrend] {n}/{len(tickers)} tickers, {len(sig)} signals so far", file=sys.stderr)
    return pd.DataFrame(sig)


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Identical engine to research/g1_ab.py / research/portfolio_backtest.py — trailing
    exit, sector-capped position sizing, tier-then-depth pool ordering, SPY>200-SMA regime
    filter. Only which signals are IN `sig` differs between variants."""
    d = sig.assign(_tr=sig.tier.map(TIER_ORDER).fillna(2))
    by_date = {dt: g.sort_values(["_tr", "depth"]) for dt, g in d.groupby("date")}
    frict = SLIP_BPS / 10000.0
    cash, positions, eq, closed = INITIAL, {}, [], []
    for dt in calendar:
        for t in list(positions):
            if dt not in bars[t].index:
                continue
            p = positions[t]
            b = bars[t].loc[dt]
            hi, lo, cl = float(b["High"]), float(b["Low"]), float(b["Close"])
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
                               "sector": p["sector"]})
                del positions[t]
        mtm = cash + sum(pp["shares"] * float(bars[tt].loc[dt, "Close"])
                         for tt, pp in positions.items() if dt in bars[tt].index)
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
                                "peak": entry, "active": False, "held": 0, "sector": s["sector"]}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def stat(res, spy, lo, hi):
    e = res["eq"][(res["eq"].index >= lo) & (res["eq"].index < hi)]["equity"]
    if len(e) < 5:
        return None
    e = e / e.iloc[0] * INITIAL
    tr = res["tr"]
    t = tr[(pd.to_datetime(tr.exit_date) >= lo) & (pd.to_datetime(tr.exit_date) < hi)] if len(tr) else tr
    r = t["r"] if len(t) else pd.Series(dtype=float)
    dd = (e / e.cummax() - 1).min()
    rr = e.pct_change().dropna()
    sp = spy.reindex(e.index).ffill().bfill()
    mcl = cur = 0
    for x in (t.sort_values("exit_date")["r"].values <= 0) if len(t) else []:
        cur = cur + 1 if x else 0
        mcl = max(mcl, cur)
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    return dict(ret=e.iloc[-1] / INITIAL - 1, dd=dd,
               sharpe=rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan,
               spy_ret=sp.iloc[-1] / sp.iloc[0] - 1, n=len(t),
               win=(r > 0).mean() * 100 if len(r) else np.nan,
               avgR=r.mean() if len(r) else np.nan, pf=pf, mcl=mcl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if SIG_CACHE.exists() and not a.rebuild:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[curtrend] loaded {len(sig)} cached signals", file=sys.stderr)
    else:
        sig = build_signals(settings)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[curtrend] built + cached {len(sig)} signals", file=sys.stderr)

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    L = ["# Current-trend gate (20-session EMA200 slope) — isolated portfolio A/B (RESEARCH)", "",
         "Tests whether requiring the CURRENT 20-session EMA200 slope not be sharply negative, "
         "ON TOP OF the existing 126-session G1 gate, helps, hurts, or washes — motivated by a "
         "live finding (2026-09-11) that 24 of 26 real screener matches had G1 reading "
         "\"uptrend\" while the current slope had already rolled over.",
         f"- signal set: current live gate (G1>=5, G2[-20,3], G3<=20, G4<=-4, weak-RR dropped) "
         f"held exactly as-is; {len(sig)} signals total.",
         "- engine: portfolio_backtest.py / g1_ab.py conventions; pool ordered knife-tier then depth", ""]

    counts = {name: int((sig.slope20 >= thr).sum()) if thr is not None else len(sig)
              for name, thr in SLOPE_VARIANTS.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")

    res = {
        name: run(sig if thr is None else sig[sig.slope20 >= thr], bars, spy_above, calendar, settings)
        for name, thr in SLOPE_VARIANTS.items()
    }

    for wn, lo, hi in windows:
        L += [f"## {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in SLOPE_VARIANTS:
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
    print(f"\n[curtrend] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
