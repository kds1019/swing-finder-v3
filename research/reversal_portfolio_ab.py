"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Portfolio-level A/B of the current V3 screen+rank against stabilisation-aware
variants, on the cached-bar universe, with an out-of-sample split.

Conventions copied from research/portfolio_backtest.py so numbers are comparable:
  entry = signal bar close (+5bps), resolution next bar
  exit  = core.trade_plan trailing stop (+2R activate / peak-1R), fib target ceiling,
          30-bar max hold
  sizing = risk_per_trade_pct of equity / (entry-stop), capped at 20% of equity and cash
  max 6 concurrent, 3 per sector, SPY>200SMA regime gate to open

Variants:
  A  current V3            gates = detect_pullback_reversal(); rank deepest pullback first
  B  gates trimmed         drop G1 (EMA200 slope) + G3 (15d range); keep G2 band + G4 VAH;
                           rank deepest first
  C  B + stabilisation FILTER   require: days_since_low>=3 AND higher_low>0 AND
                           close_vs_ema20>-6 AND last_5d_return>-4   (still deepest-first)
  D  B + stabilisation RANK     no filter; order stabilising-then-deepest, still_falling last
  E  C + knife-averse RANK      stabilisation filter; order shallowest / most-stabilised first

Splits: full, 2021-2024 (train), 2025-2026 (test), and 2022 alone.

Usage:  python -m research.reversal_portfolio_ab
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    MIN_BARS_FOR_SCREENER, EMA200_MIN_UPTREND_PCT,
    PRICE_VS_EMA200_MAX_PCT, PRICE_VS_EMA200_MIN_PCT,
    MAX_PRICE_VS_VALUE_AREA_HIGH_PCT, CONSOLIDATION_MAX_RANGE_PCT,
    measure_pullback_reversal,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
OUT = Path(__file__).resolve().parent / "reversal_portfolio_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT = 20.0
MAX_POS = 6
SECTOR_CAP = 3
SLIP_BPS = 5
MAX_HOLD = 30
WINDOW = 300
START = "2021-06-01"
VARIANTS = ["A", "B", "C", "D", "E"]


def _stab_fields(df: pd.DataFrame, i: int) -> dict:
    """cheap stabilisation read at bar i using bars <= i (mirrors
    core.pullback_reversal.measure_stabilization + a couple of extras)."""
    lo = df["Low"].values
    cl = df["Close"].values
    e20 = df["EMA20"].values
    a = max(0, i - 19)
    win_lo = lo[a: i + 1]
    wl = float(win_lo.min())
    dsl = len(win_lo) - 1 - int(np.argmin(win_lo))
    rec3 = float(lo[max(0, i - 2): i + 1].min())
    hl = (rec3 - wl) / wl * 100 if wl > 0 else 0.0
    cvse20 = (cl[i] / e20[i] - 1) * 100 if e20[i] > 0 else 0.0
    r5 = (cl[i] / cl[i - 5] - 1) * 100 if i >= 5 else 0.0
    return {"days_since_low": dsl, "higher_low_pct": hl,
            "close_vs_ema20_pct": cvse20, "last_5d_return_pct": r5}


def build_signals(tickers, sectors, settings) -> pd.DataFrame:
    sig = []
    for n, t in enumerate(tickers, 1):
        cache = BARS_DIR / f"{t}.pkl"
        if not cache.exists():
            continue
        raw = pd.read_pickle(cache)
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        rng = (df["Close"].rolling(15).max() - df["Close"].rolling(15).min()) / df["Close"] * 100.0
        # wide net: rising 200EMA + roughly in the pullback zone (loose, so every
        # variant's own gates decide inclusion)
        net = (upt > 0) & pvs.between(-25, 8)
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1:
                continue
            d = df["Date"].iloc[i]
            if str(d.date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            m = measure_pullback_reversal(prefix)          # one volume-profile compute
            if m is None or not m["volume_profile_available"]:
                continue
            vah_pct = m["price_vs_value_area_high_pct"]
            m_pvs, m_upt, m_rng = m["price_vs_ema200_pct"], m["ema200_uptrend_pct"], m["consolidation_range_pct"]
            # current live gates, inline (== detect_pullback_reversal verdict)
            detected = (m_upt >= EMA200_MIN_UPTREND_PCT
                        and PRICE_VS_EMA200_MIN_PCT <= m_pvs <= PRICE_VS_EMA200_MAX_PCT
                        and m_rng <= CONSOLIDATION_MAX_RANGE_PCT
                        and vah_pct is not None and vah_pct <= MAX_PRICE_VS_VALUE_AREA_HIGH_PCT)
            plan = compute_trade_plan(prefix, settings)
            if plan is None or plan["stop"] >= plan["entry"] or plan["weak_rr"]:
                continue
            st = _stab_fields(df, i)
            stabilising = (st["days_since_low"] >= 3 and st["higher_low_pct"] > 0
                           and st["close_vs_ema20_pct"] > -6 and st["last_5d_return_pct"] > -4)
            sig.append({
                "date": d, "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "pvs": m_pvs, "upt": m_upt, "rng": m_rng, "vah_pct": vah_pct,
                "detected": bool(detected), "stabilising": stabilising,
                "days_since_low": st["days_since_low"],
            })
        if n % 120 == 0:
            print(f"[ab] signals {n}/{len(tickers)}  ({len(sig)})", file=sys.stderr)
    return pd.DataFrame(sig)


def eligible(sig: pd.DataFrame, variant: str) -> pd.DataFrame:
    d = sig
    if variant == "A":
        return d[d.detected]
    # B/C/D/E share the trimmed gate set: G2 band + G4 VAH (no G1 slope, no G3 range)
    base = d[d.pvs.between(PRICE_VS_EMA200_MIN_PCT, PRICE_VS_EMA200_MAX_PCT)
             & (d.vah_pct.notna()) & (d.vah_pct <= MAX_PRICE_VS_VALUE_AREA_HIGH_PCT)]
    if variant in ("C", "E"):
        base = base[base.stabilising]
    return base


def rank_key(g: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant in ("A", "B", "C"):
        return g.sort_values("pvs")                       # deepest pullback first
    if variant == "D":
        return g.sort_values(["stabilising", "pvs"], ascending=[False, True])
    if variant == "E":
        return g.sort_values(["stabilising", "pvs"], ascending=[False, False])  # shallowest first
    return g


def run_variant(sig: pd.DataFrame, bars: dict, spy_above: pd.Series, sectors, settings,
                variant: str, calendar) -> dict:
    elig = eligible(sig, variant)
    by_date = {d: g for d, g in elig.groupby("date")}
    frict = SLIP_BPS / 10000.0
    cash = INITIAL
    positions: dict[str, dict] = {}
    eq, closed = [], []

    for d in calendar:
        for t in list(positions.keys()):
            p = positions[t]
            if d not in bars[t].index:
                continue
            bar = bars[t].loc[d]
            hi, lo, cl = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
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
                closed.append({"ticker": t, "exit_date": d, "reason": xr,
                               "r": (xp - p["entry"]) / risk, "sector": p["sector"],
                               "held": p["held"]})
                del positions[t]

        mtm = cash + sum(pp["shares"] * float(bars[tt].loc[d, "Close"])
                         for tt, pp in positions.items() if d in bars[tt].index)
        eq.append((d, mtm))

        if d in by_date and bool(spy_above.get(d, False)) and len(positions) < MAX_POS:
            sec = {}
            for pp in positions.values():
                sec[pp["sector"]] = sec.get(pp["sector"], 0) + 1
            for _, s in rank_key(by_date[d], variant).iterrows():
                t = s["ticker"]
                if t in positions or len(positions) >= MAX_POS:
                    continue
                if sec.get(s["sector"], 0) >= SECTOR_CAP:
                    continue
                entry = s["entry"] * (1 + frict)
                rps = entry - s["stop"]
                if rps <= 0:
                    continue
                shares = int(min((mtm * settings.risk_per_trade_pct / 100) / rps,
                                 (mtm * MAX_POS_PCT / 100) / entry, cash / entry))
                if shares <= 0:
                    continue
                cash -= shares * entry
                positions[t] = {"shares": shares, "entry": entry, "stop": s["stop"],
                                "target": s["target"], "peak": entry, "active": False,
                                "held": 0, "sector": s["sector"]}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1

    last = calendar[-1]
    for t, p in list(positions.items()):
        if last in bars[t].index:
            cl = float(bars[t].loc[last, "Close"])
            cash += p["shares"] * cl * (1 - frict)
            closed.append({"ticker": t, "exit_date": last, "reason": "open_end",
                           "r": (cl - p["entry"]) / (p["entry"] - p["stop"]),
                           "sector": p["sector"], "held": p["held"]})

    eqs = pd.DataFrame(eq, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(closed)
    return {"eq": eqs, "tr": tr, "n_signals": len(elig)}


def stats(eqs: pd.DataFrame, tr: pd.DataFrame, spy: pd.Series, lo: str, hi: str) -> dict:
    e = eqs[(eqs.index >= lo) & (eqs.index < hi)]["equity"]
    t = tr[(pd.to_datetime(tr.exit_date) >= lo) & (pd.to_datetime(tr.exit_date) < hi)] if len(tr) else tr
    if len(e) < 5:
        return {}
    e = e / e.iloc[0] * INITIAL
    ret = e.iloc[-1] / INITIAL - 1
    dd = (e / e.cummax() - 1).min()
    rr = e.pct_change().dropna()
    sharpe = rr.mean() / rr.std() * np.sqrt(252) if rr.std() > 0 else np.nan
    sp = spy.reindex(e.index).ffill().bfill()
    sp_ret = sp.iloc[-1] / sp.iloc[0] - 1
    sp_dd = (sp / sp.cummax() - 1).min()
    r = t["r"] if len(t) else pd.Series(dtype=float)
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    # max consecutive losing trades in date order
    mcl = cur = 0
    for x in (t.sort_values("exit_date")["r"].values <= 0) if len(t) else []:
        cur = cur + 1 if x else 0
        mcl = max(mcl, cur)
    reasons = t["reason"].value_counts().to_dict() if len(t) else {}
    sec_max = (t["sector"].value_counts().iloc[0] / len(t) * 100) if len(t) else np.nan
    return {
        "ret": ret, "spy_ret": sp_ret, "dd": dd, "spy_dd": sp_dd, "sharpe": sharpe,
        "trades": len(t), "win": (r > 0).mean() * 100 if len(r) else np.nan,
        "avgR": r.mean() if len(r) else np.nan, "medR": r.median() if len(r) else np.nan,
        "pf": pf, "mcl": mcl, "reasons": reasons, "sec_max_pct": sec_max,
    }


def fmtrow(name: str, s: dict) -> str:
    if not s:
        return f"| {name} | (insufficient) | | | | | | | | |"
    return (f"| {name} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
            f"{s['trades']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} | "
            f"{s['sec_max_pct']:.0f}% |")


def main() -> None:
    settings = load_settings()
    uni = build_universe(settings)
    sectors = dict(zip(uni["Ticker"], uni["Sector"]))
    cached = sorted(p.stem for p in BARS_DIR.glob("*.pkl") if p.stem != "SPY")
    tickers = [t for t in cached if t in sectors]
    print(f"[ab] {len(tickers)} tickers", file=sys.stderr)

    bars = {}
    for t in tickers:
        bars[t] = pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date")
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()

    sig = build_signals(tickers, sectors, settings).sort_values("date").reset_index(drop=True)
    print(f"[ab] {len(sig)} wide-net signals", file=sys.stderr)
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"),
               ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"),
               ("2022 only", "2022-01-01", "2023-01-01")]

    L = ["# Reversal portfolio A/B (RESEARCH)", "",
         "Same engine/costs as research/portfolio_backtest.py. Columns: total return, "
         "max DD, Sharpe, #trades, win%, avgR, PF, max consecutive losers, largest single-sector "
         "share of trades.", ""]
    res = {}
    for v in VARIANTS:
        res[v] = run_variant(sig, bars, spy_above, sectors, settings, v, calendar)
        print(f"[ab] variant {v}: {res[v]['n_signals']} eligible signals, "
              f"{len(res[v]['tr'])} trades", file=sys.stderr)

    for wname, lo, hi in windows:
        L += [f"## {wname}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss | sectorMax |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        sp0 = stats(res["A"]["eq"], res["A"]["tr"], spy, lo, hi)
        for v in VARIANTS:
            s = stats(res[v]["eq"], res[v]["tr"], spy, lo, hi)
            L.append(fmtrow(f"{v}", s))
        if sp0:
            L.append(f"| SPY | {sp0['spy_ret']*100:+.0f}% | {sp0['spy_dd']*100:.0f}% | | | | | | | |")
        L.append("")
        for v in VARIANTS:
            s = stats(res[v]["eq"], res[v]["tr"], spy, lo, hi)
            if s:
                L.append(f"- {v} exit reasons ({wname}): {s['reasons']}")
        L.append("")

    L += ["## Signal counts (eligible, full period)", ""]
    for v in VARIANTS:
        L.append(f"- {v}: {res[v]['n_signals']} eligible signals over the whole window")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ab] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
