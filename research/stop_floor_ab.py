"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Isolated test of initial-stop placement. One question: at the portfolio level, out of
sample, is the pipeline better off with the CURRENT stop (core.trade_plan's
nearest-support refinement, which can pull the stop to within a fraction of an ATR of
price) or a stop with a minimum distance?

Motivation (live pick log, 2026-09-30): 23 of 25 resolved picks hit their stop, 16 on the
very next bar, median stop distance 2.15% (NJR 0.47%, FNB 0.57%, CECO 0.91%); the only two
winners had 4.0% / 6.9% stops.

Variants (all via compute_trade_plan's research knobs; "base" == live behavior exactly):
  base          current live stop
  no_tighten    support refinement may only lower the stop, never raise it
  floor_0.75    stop never closer than 0.75 * ATR14
  floor_1.0     ... 1.0 * ATR14
  floor_1.5     ... 1.5 * ATR14
  nt_floor_1.0  no_tighten + 1.0 ATR floor

Two candidate-set framings, reported separately:
  isolated  the candidate set is exactly what the live screener passes (current gates +
            the base plan's weak-RR drop), and ONLY the stop/target come from the variant.
            Answers "same trades, different stop".
  gated     each variant's own plan also decides the weak-RR drop (a wider stop lowers
            R:R, so fewer candidates survive). Answers "what would the live pipeline
            actually trade if this shipped as-is".

Engine: identical to research/weak_rr_ab.py (portfolio_backtest.py conventions — 6 positions,
3/sector, 20% max position, 5bps slippage, SPY>200SMA gate, +2R/1R trailing exit, 30-bar
max hold, pool ordered knife-risk tier then depth).

Bars: fetched via research/build_calibration_dataset.get_history into research/data/bars
(cached, gitignored). Signals cached to research/data/stop_floor_signals.pkl.

Usage:  python -m research.stop_floor_ab --limit 500          (fetch + build + report)
        python -m research.stop_floor_ab --rebuild            (force re-scan of signals)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    MIN_BARS_FOR_SCREENER, classify_knife_risk, detect_pullback_reversal, measure_stabilization,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "stop_floor_signals.pkl"
OUT = Path(__file__).resolve().parent / "stop_floor_ab.md"
START = "2021-06-01"

# Portfolio engine constants + run()/stat() below are copied verbatim from
# research/weak_rr_ab.py (same conventions as every prior isolated A/B) rather than
# imported: every existing *_ab.py script currently fails to import (it references
# core.pullback_reversal.PRICE_VS_EMA200_MAX_PCT, since removed from that module).
INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD, WINDOW = 20.0, 6, 3, 5, 30, 300
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

VARIANTS: dict[str, dict] = {
    "base":         {},
    "no_tighten":   {"support_can_tighten": False},
    "floor_0.75":   {"min_stop_atr": 0.75},
    "floor_1.0":    {"min_stop_atr": 1.0},
    "floor_1.5":    {"min_stop_atr": 1.5},
    "nt_floor_1.0": {"support_can_tighten": False, "min_stop_atr": 1.0},
}


def fetch_bars(settings, limit: int, seed: int, start: str) -> list[str]:
    """Populate BARS_DIR for a seeded random sample of the live universe (+ SPY)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from research.build_calibration_dataset import get_history

    client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
    uni = build_universe(settings)
    tickers = uni["Ticker"].tolist()
    if limit:
        rng = np.random.default_rng(seed)
        tickers = list(rng.choice(tickers, size=min(limit, len(tickers)), replace=False))
    st = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    get_history(client, "SPY", st)
    t0 = time.time()
    for n, t in enumerate(sorted(tickers), 1):
        get_history(client, t, st)
        if n % 50 == 0:
            print(f"[sf] fetched {n}/{len(tickers)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return sorted(t for t in tickers if t != "SPY")


def build_signals(settings, tickers: list[str], sectors: dict) -> pd.DataFrame:
    sig = []
    t0 = time.time()
    for n, t in enumerate(tickers, 1):
        path = BARS_DIR / f"{t}.pkl"
        if not path.exists():
            continue
        raw = pd.read_pickle(path)
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        df = compute_indicators(raw.copy())
        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        net = (upt > 0) & pvs.between(-25, 8)  # cheap prefilter; the real gates run below
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1 or str(df["Date"].iloc[i].date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            det = detect_pullback_reversal(prefix)
            if not det.get("detected"):
                continue
            atr = float(prefix["ATR14"].iloc[-1])
            row = {"date": df["Date"].iloc[i], "ticker": t, "sector": sectors.get(t, "Unknown"),
                   "pvs": det["price_vs_ema200_pct"],
                   "tier": classify_knife_risk(measure_stabilization(prefix)),
                   "atr": atr}
            ok = False
            for v, kw in VARIANTS.items():
                p = compute_trade_plan(prefix, settings, **kw)
                if p is None or p["stop"] >= p["entry"]:
                    row[f"{v}_stop"] = np.nan
                    continue
                ok = True
                row["entry"] = p["entry"]
                row[f"{v}_stop"] = p["stop"]
                row[f"{v}_target"] = p["target"]
                row[f"{v}_rr"] = p["rr_ratio"]
                row[f"{v}_weak"] = bool(p["weak_rr"])
            if ok:
                sig.append(row)
        if n % 50 == 0:
            print(f"[sf] scanned {n}/{len(tickers)} ({len(sig)} signals, {time.time()-t0:.0f}s)",
                  file=sys.stderr)
    return pd.DataFrame(sig)


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Mark-to-market carries forward each held position's last KNOWN close on a day its
    own bar is missing, rather than dropping it from the day's total entirely — see
    research/current_trend_gate_ab.py::run() for why (a real, confirmed gap in the free IEX
    feed on 2021-10-25 for ~231 of ~480 cached tickers previously produced a fake one-day
    ~20-40% portfolio "crash" that fully reversed the next real bar)."""
    d = sig.assign(_tr=sig.tier.map(TIER_ORDER).fillna(2))
    by_date = {dt: g.sort_values(["_tr", "pvs"]) for dt, g in d.groupby("date")}
    frict = SLIP_BPS / 10000.0
    cash, positions, eq, closed = INITIAL, {}, [], []
    for dt in calendar:
        for t in list(positions):
            p = positions[t]
            if dt not in bars[t].index:
                continue  # no bar today -- can't check exits; p["last_close"] carries forward
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
                               "sector": p["sector"], "weak_rr": p["weak_rr"]})
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
                                "weak_rr": bool(s["weak_rr"]), "last_close": entry}
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


def variant_frame(sig: pd.DataFrame, v: str, framing: str) -> pd.DataFrame:
    d = sig.dropna(subset=["base_stop", f"{v}_stop"])
    gate_col = "base_weak" if framing == "isolated" else f"{v}_weak"
    d = d[~d[gate_col].astype(bool)]
    return pd.DataFrame({
        "date": d["date"], "ticker": d["ticker"], "sector": d["sector"], "entry": d["entry"],
        "stop": d[f"{v}_stop"], "target": d[f"{v}_target"], "rr_ratio": d[f"{v}_rr"],
        "weak_rr": False, "pvs": d["pvs"], "tier": d["tier"],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="use whatever is already in BARS_DIR")
    a = ap.parse_args()
    settings = load_settings()

    if SIG_CACHE.exists() and not a.rebuild:
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[sf] loaded {len(sig)} cached signals", file=sys.stderr)
    else:
        if a.no_fetch:
            tickers = sorted(p.stem for p in BARS_DIR.glob("*.pkl") if p.stem != "SPY")
        else:
            tickers = fetch_bars(settings, a.limit, a.seed, a.start)
        uni = build_universe(settings)
        sectors = dict(zip(uni["Ticker"], uni["Sector"]))
        sig = build_signals(settings, tickers, sectors)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[sf] built + cached {len(sig)} signals", file=sys.stderr)

    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    tickers = sorted(sig.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    live = sig[~sig["base_weak"].astype(bool)]
    L = ["# Initial-stop floor — isolated A/B (RESEARCH)", "",
         f"- {len(tickers)} tickers, {len(sig)} screener detections, "
         f"{len(live)} pass the live weak-RR gate",
         f"- engine: run()/stat() copied verbatim from research/weak_rr_ab.py; start {START}",
         "- isolated = live candidate set, only the stop/target differ | "
         "gated = each variant's own weak-RR drop also applies", "",
         "## Stop geometry on the live candidate set", "",
         "| variant | median stop % | p10 stop % | median stop/ATR | share < 1 ATR | weak-RR under variant |",
         "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        d = live.dropna(subset=[f"{v}_stop"])
        sp = (d["entry"] - d[f"{v}_stop"]) / d["entry"] * 100
        sa = (d["entry"] - d[f"{v}_stop"]) / d["atr"]
        L.append(f"| {v} | {sp.median():.2f}% | {sp.quantile(.1):.2f}% | {sa.median():.2f} | "
                 f"{(sa < 1).mean():.0%} | {d[f'{v}_weak'].astype(bool).mean():.0%} |")
    L.append("")

    for framing in ("isolated", "gated"):
        res = {v: run(variant_frame(sig, v, framing), bars, spy_above, calendar, settings)
               for v in VARIANTS}
        for wn, lo, hi in windows:
            L += [f"## {framing} — {wn}", "",
                  "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
                  "|---|---|---|---|---|---|---|---|---|"]
            sp0 = None
            for v in VARIANTS:
                s = stat(res[v], spy, lo, hi)
                if s is None:
                    L.append(f"| {v} | (insufficient) | | | | | | | |")
                    continue
                sp0 = s["spy_ret"]
                L.append(f"| {v} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                         f"{s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
            if sp0 is not None:
                L.append(f"| SPY | {sp0*100:+.0f}% | | | | | | | |")
            L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[sf] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
