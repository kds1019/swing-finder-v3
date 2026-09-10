"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated test of weak-RR handling. One question: at the portfolio level, out of
sample, is the pipeline better off KEEPING weak-RR signals, DROPPING them, or
RE-FLOORING them (tighten the stop so R:R can't fall below settings.min_risk_reward,
i.e. don't let the support-cluster refinement cross the floor)?

Everything else is held at the CURRENT merged screener: G1-G4 gates unchanged,
candidate pool ordered knife-risk tier then depth (matching main after PR #62),
same trailing exit / sizing / 6-position / 3-sector / SPY>200SMA engine as
research/portfolio_backtest.py.

Signals are built once (weak-RR kept and flagged) and cached to
research/data/weak_rr_signals.pkl so re-runs are instant.

Splits: full, 2021-2024 (train), 2025-2026 (test), 2022 alone.

Usage:  python -m research.weak_rr_ab            (uses cache if present)
        python -m research.weak_rr_ab --rebuild  (force re-scan)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    MIN_BARS_FOR_SCREENER, EMA200_MIN_UPTREND_PCT, PRICE_VS_EMA200_MAX_PCT,
    PRICE_VS_EMA200_MIN_PCT, MAX_PRICE_VS_VALUE_AREA_HIGH_PCT, CONSOLIDATION_MAX_RANGE_PCT,
    measure_pullback_reversal, classify_knife_risk,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "weak_rr_signals.pkl"
OUT = Path(__file__).resolve().parent / "weak_rr_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD, WINDOW = 20.0, 6, 3, 5, 30, 300
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}


def _stab(df, i):
    lo, cl, e20 = df["Low"].values, df["Close"].values, df["EMA20"].values
    a = max(0, i - 19)
    wl = float(lo[a:i + 1].min())
    dsl = len(lo[a:i + 1]) - 1 - int(np.argmin(lo[a:i + 1]))
    hl = (float(lo[max(0, i - 2):i + 1].min()) - wl) / wl * 100 if wl > 0 else 0.0
    cvs = (cl[i] / e20[i] - 1) * 100 if e20[i] > 0 else 0.0
    r5 = (cl[i] / cl[i - 5] - 1) * 100 if i >= 5 else 0.0
    return {"days_since_pullback_low": dsl, "higher_low_pct": hl,
            "close_vs_ema20_pct": cvs, "last_5d_return_pct": r5}


def build_signals(settings) -> pd.DataFrame:
    uni = build_universe(settings)
    sectors = dict(zip(uni["Ticker"], uni["Sector"]))
    tickers = sorted(p.stem for p in BARS_DIR.glob("*.pkl")
                     if p.stem != "SPY" and p.stem in sectors)
    print(f"[wk] {len(tickers)} tickers", file=sys.stderr)
    sig = []
    for n, t in enumerate(tickers, 1):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        net = (upt > 0) & pvs.between(-25, 8)
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1 or str(df["Date"].iloc[i].date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            m = measure_pullback_reversal(prefix)
            if m is None or not m["volume_profile_available"]:
                continue
            vah = m["price_vs_value_area_high_pct"]
            pv, up, rg = m["price_vs_ema200_pct"], m["ema200_uptrend_pct"], m["consolidation_range_pct"]
            detected = (up >= EMA200_MIN_UPTREND_PCT
                        and PRICE_VS_EMA200_MIN_PCT <= pv <= PRICE_VS_EMA200_MAX_PCT
                        and rg <= CONSOLIDATION_MAX_RANGE_PCT
                        and vah is not None and vah <= MAX_PRICE_VS_VALUE_AREA_HIGH_PCT)
            if not detected:
                continue
            plan = compute_trade_plan(prefix, settings)
            if plan is None or plan["stop"] >= plan["entry"]:
                continue
            st = _stab(df, i)
            sig.append({
                "date": df["Date"].iloc[i], "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "rr_ratio": plan["rr_ratio"], "weak_rr": bool(plan["weak_rr"]),
                "pvs": pv, "tier": classify_knife_risk(st),
            })
        if n % 120 == 0:
            print(f"[wk] {n}/{len(tickers)} ({len(sig)})", file=sys.stderr)
    return pd.DataFrame(sig)


def apply_mode(sig: pd.DataFrame, mode: str, min_rr: float) -> pd.DataFrame:
    d = sig.copy()
    if mode == "keep":
        return d
    if mode == "drop":
        return d[~d.weak_rr]
    if mode == "refloor":
        # tighten the stop on weak-RR plans so R:R == min_rr exactly (don't let the
        # support refinement cross the floor). target unchanged.
        w = d.weak_rr & (d.target > d.entry)
        d.loc[w, "stop"] = d.loc[w, "entry"] - (d.loc[w, "target"] - d.loc[w, "entry"]) / min_rr
        d.loc[w, "rr_ratio"] = min_rr
        d.loc[w, "weak_rr"] = False
        return d[d.stop < d.entry]
    raise ValueError(mode)


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    d = sig.assign(_tr=sig.tier.map(TIER_ORDER).fillna(2))
    by_date = {dt: g.sort_values(["_tr", "pvs"]) for dt, g in d.groupby("date")}
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
                               "sector": p["sector"], "weak_rr": p["weak_rr"]})
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
                                "peak": entry, "active": False, "held": 0, "sector": s["sector"],
                                "weak_rr": bool(s["weak_rr"])}
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
               avgR=r.mean() if len(r) else np.nan, pf=pf, mcl=mcl,
               nweak=int(t["weak_rr"].sum()) if len(t) else 0)


def tl_row(label, g):
    r = g["r_multiple"] if "r_multiple" in g else g["r"]
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return f"  {label:<22} n={len(g):>6} win={100*(r>0).mean():>4.0f}% avgR={r.mean():>+5.2f} PF={pos/neg if neg>0 else 9.9:>4.2f}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    settings = load_settings()

    if SIG_CACHE.exists() and not a.rebuild:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[wk] loaded {len(sig)} cached signals", file=sys.stderr)
    else:
        sig = build_signals(settings)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[wk] built + cached {len(sig)} signals", file=sys.stderr)

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]
    modes = ["keep", "drop", "refloor"]

    L = ["# Weak-RR handling — isolated A/B (RESEARCH)", "",
         f"- signal set: current merged screener (G1-G4), {len(sig)} detected signals, "
         f"weak-RR share {sig.weak_rr.mean():.0%}",
         "- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth",
         "- keep = all signals | drop = exclude weak_rr | refloor = tighten weak_rr stop to exactly min_rr:1",
         ""]

    res = {m: run(apply_mode(sig, m, settings.min_risk_reward), bars, spy_above, calendar, settings)
           for m in modes}

    for wn, lo, hi in windows:
        L += [f"## {wn}", "",
              "| mode | ret | maxDD | Sharpe | trades | (weak) | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for m in modes:
            s = stat(res[m], spy, lo, hi)
            if s is None:
                L.append(f"| {m} | (insufficient) | | | | | | | | |"); continue
            sp0 = s["spy_ret"]
            L.append(f"| {m} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                     f"{s['n']} | {s['nweak']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
        if sp0 is not None:
            L.append(f"| SPY | {sp0*100:+.0f}% | | | | | | | | |")
        L.append("")

    # trade-level split (all detected signals, resolved with the live trailing exit already in
    # the calibration dataset if available; else from the sim's closed trades under 'keep')
    L += ["## Trade-level: weak-RR vs non-weak (sim 'keep' closed trades)", ""]
    kt = res["keep"]["tr"].copy()
    kt["yr"] = pd.to_datetime(kt["exit_date"]).dt.year
    for split, mask in [("2021-2024", kt.yr <= 2024), ("2025-2026", kt.yr >= 2025), ("2022", kt.yr == 2022)]:
        s = kt[mask]
        L.append(f"_{split}_")
        L.append(tl_row("weak_rr", s[s.weak_rr]))
        L.append(tl_row("non-weak", s[~s.weak_rr]))
        L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[wk] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
