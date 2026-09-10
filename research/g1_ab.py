"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated portfolio test of the G1 gate (EMA200 6-month slope >= EMA200_MIN_UPTREND_PCT,
currently 5%). Question: at the portfolio level, out of sample, does lowering G1 to
>=3 or >=0 help, hurt, or wash?

G2 (price -20..+3 vs EMA200), G3 (15d range <= 20), G4 (VAH <= -4) held at current
values. weak-RR dropped (matching the direction of PR #63). Pool ordered knife-risk
tier then depth. Same engine/costs as research/portfolio_backtest.py.

Signals built once (G1 relaxed to a wide net, ema200_uptrend_pct kept as a column)
and cached to research/data/g1_signals.pkl.

Splits: full, 2021-2024 (train), 2025-2026 (test), 2022 alone.

Usage:  python -m research.g1_ab            (uses cache if present)
        python -m research.g1_ab --rebuild
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
    MAX_PRICE_VS_VALUE_AREA_HIGH_PCT, CONSOLIDATION_MAX_RANGE_PCT,
    measure_pullback_reversal, classify_knife_risk,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "g1_signals.pkl"
OUT = Path(__file__).resolve().parent / "g1_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD, WINDOW = 20.0, 6, 3, 5, 30, 300
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}
G1_VARIANTS = {"G1>=5 (current)": 5.0, "G1>=3": 3.0, "G1>=0": 0.0}


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
    print(f"[g1] {len(tickers)} tickers", file=sys.stderr)
    sig = []
    for n, t in enumerate(tickers, 1):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        # wide net: G1 relaxed to >=-3 (margin below 0), G2 band held
        net = (upt >= -3) & pvs.between(PRICE_VS_EMA200_MIN_PCT, PRICE_VS_EMA200_MAX_PCT)
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1 or str(df["Date"].iloc[i].date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            m = measure_pullback_reversal(prefix)
            if m is None or not m["volume_profile_available"]:
                continue
            vah = m["price_vs_value_area_high_pct"]
            rg = m["consolidation_range_pct"]
            # G3 + G4 held at current values (G1 varied downstream, G2 already in the net)
            if rg > CONSOLIDATION_MAX_RANGE_PCT or vah is None or vah > MAX_PRICE_VS_VALUE_AREA_HIGH_PCT:
                continue
            try:
                plan = compute_trade_plan(prefix, settings)
            except ZeroDivisionError:
                # core.trade_plan.find_support_resistance.cluster() can divide by a ~0
                # pivot price on deep-split-adjusted ancient bars. Live scans never hit this
                # (price_min = $10); the wide research net replaying full history does. Skip.
                continue
            if plan is None or plan["stop"] >= plan["entry"] or plan["weak_rr"]:
                continue  # weak-RR dropped (PR #63 direction)
            st = _stab(df, i)
            sig.append({
                "date": df["Date"].iloc[i], "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "upt": m["ema200_uptrend_pct"], "pvs": m["price_vs_ema200_pct"],
                "tier": classify_knife_risk(st),
            })
        if n % 120 == 0:
            print(f"[g1] {n}/{len(tickers)} ({len(sig)})", file=sys.stderr)
    return pd.DataFrame(sig)


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
        print(f"[g1] loaded {len(sig)} cached signals", file=sys.stderr)
    else:
        sig = build_signals(settings)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[g1] built + cached {len(sig)} signals", file=sys.stderr)

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    L = ["# G1 (EMA200 slope) — isolated portfolio A/B (RESEARCH)", "",
         f"- signal set: G2[-20,3] + G3<=20 + G4<=-4 held, weak-RR dropped, G1 relaxed. "
         f"{len(sig)} signals total; G1>=5 covers {int((sig.upt>=5).sum())}, "
         f">=3 {int((sig.upt>=3).sum())}, >=0 {int((sig.upt>=0).sum())}.",
         "- engine: portfolio_backtest.py conventions; pool ordered knife-tier then depth", ""]

    res = {name: run(sig[sig.upt >= thr], bars, spy_above, calendar, settings)
           for name, thr in G1_VARIANTS.items()}

    for wn, lo, hi in windows:
        L += [f"## {wn}", "",
              "| G1 | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in G1_VARIANTS:
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
    print(f"\n[g1] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
