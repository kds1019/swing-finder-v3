"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Does the candidate-pool ORDER help? agents/market_data_agent.scan_universe orders screener
matches knife-risk tier first (stabilising -> forming -> still_falling), deepest pullback
within a tier. Its own docstring calls that "a pool-ordering heuristic ... NOT a validated
ranking". Motivation (research/harmonic_ab.md, 2026-10-17): on the live signal set,
signals squeezed OUT of the top-30 pool did better per trade than those kept (+0.19R vs
+0.13R), and episode-first still_falling signals had the best avgR of the three tiers.

In this portfolio engine the order decides which signals take the free slots of the
6-position book — the same role the pool order plus the Decision Agent's
support_status-first ranking play live. Variants (fixed before the run):

  base           live order: tier (stabilising, forming, unknown, still_falling), then depth
  depth_only     deepest pullback first, tier ignored
  falling_first  tier reversed (still_falling, forming, unknown, stabilising), then depth
  random         seeded random order — control: how much does order matter at all?

Signals: research/harmonic_ab.build_rows (screener_gate pass + the live weak-RR drop),
i.e. exactly the signal set in research/harmonic_ab.md. Engine: research/harmonic_ab.run
conventions (copied from research/stop_floor_ab.py), with the sort key made a parameter.
Noise-aware comparison via research/robust.py.

Not modelled: the Decision Agent's own judgment and research/fundamentals layer, and the
support-check downgrade (pipeline.compute_support_check) applied on top of the tier live.

Usage:  python -m research.pool_order_ab                (fetch frozen tickers + build + report)
        python -m research.pool_order_ab --no-fetch     (use cached research/data/bars)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from research.harmonic_ab import (
    BARS_DIR, EPISODE_GAP, INITIAL, MAX_HOLD, MAX_POS, MAX_POS_PCT, SECTOR_CAP, SIG_CACHE,
    SLIP_BPS, START, WINDOWS, build_rows, episode_first, rstats, stat,
)
from research.robust import frozen_universe, robust_compare, robust_markdown
from research.stop_floor_ab import fetch_bars

OUT = Path(__file__).resolve().parent / "pool_order_ab.md"
RANDOM_SEED = 11

TIERS = ("stabilising", "forming", None, "still_falling")
ORDERS = {
    "base":          {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3},
    "depth_only":    None,
    "falling_first": {"still_falling": 0, "forming": 1, None: 2, "stabilising": 3},
    "random":        "random",
}


def ranked(sig: pd.DataFrame, order) -> pd.DataFrame:
    """Attach `_rank` (ascending = taken first). Depth tie-break = most negative pvs first."""
    d = sig.copy()
    depth = d["pvs"].rank(method="first")
    if order is None:
        d["_rank"] = depth
    elif order == "random":
        d["_rank"] = np.random.default_rng(RANDOM_SEED).random(len(d))
    else:
        d["_rank"] = d["tier"].map(order).fillna(2) * 1e7 + depth
    return d


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """research/harmonic_ab.run with the pool order taken from sig['_rank']; tier carried to trades."""
    by_date = {dt: g.sort_values("_rank") for dt, g in sig.groupby("date")}
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
                               "sector": p["sector"], "weak_rr": False, "tier": p["tier"]})
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
                                "tier": s["tier"], "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="use whatever is already in BARS_DIR")
    ap.add_argument("--subsamples", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.9)
    a = ap.parse_args()
    settings = load_settings()

    frozen = frozen_universe(settings, limit=a.limit, seed=a.seed)
    sectors = dict(zip(frozen["Ticker"], frozen["Sector"]))
    if SIG_CACHE.exists() and not a.rebuild:
        rows = pd.read_pickle(SIG_CACHE)
        print(f"[pool] loaded {len(rows)} cached rows", file=sys.stderr)
    else:
        tickers = sorted(frozen["Ticker"])
        if not a.no_fetch:
            fetch_bars(settings, tickers, a.start)
        rows = build_rows(settings, tickers, sectors)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        rows.to_pickle(SIG_CACHE)
        print(f"[pool] built + cached {len(rows)} rows", file=sys.stderr)

    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["ticker"].isin(sectors)].copy()
    rows["sector"] = rows["ticker"].map(sectors)
    live = rows[(rows.gate == "pass") & (~rows.weak_rr)].copy()
    tickers = sorted(live.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    spy_above.index = pd.to_datetime(spy_above.index).normalize()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})
    live["spy_ok"] = live["date"].map(spy_above).fillna(False).astype(bool)

    frames = {k: ranked(live, o) for k, o in ORDERS.items()}
    res = {k: run(v, bars, spy_above, calendar, settings) for k, v in frames.items()}

    per_day = live[live.spy_ok].groupby("date").size()
    L = ["# Candidate-pool order — portfolio A/B (RESEARCH)", "",
         f"- {len(tickers)} tickers (research/frozen_universe.csv), {len(live)} live signals "
         f"(screener pass + weak-RR drop) from {START}",
         f"- signals per SPY>200SMA day: median {per_day.median():.0f}, "
         f"{(per_day > MAX_POS).mean():.0%} of days have more signals than the {MAX_POS}-slot book "
         "(only on those days, with a slot free, can the order matter)",
         "- order variants: " + ", ".join(ORDERS) + f" (random seed {RANDOM_SEED}); "
         "depth tie-break = deepest below EMA200 first", "",
         "## Per-signal outcome by tier (no portfolio; every SPY>200SMA signal on its own)", "",
         "| tier | all signals: n / avgR / PF | episode-first: n / avgR / PF | "
         "episode-first avgR 2021-24 / 2025-26 |", "|---|---|---|---|"]
    ok = live[live.spy_ok]
    ep = episode_first(ok)
    f = lambda s: f"{s['n']} / {s['avgR']:+.2f} / {s['pf']:.2f}" if s["n"] else "0"
    for tier in TIERS:
        g_all = ok[ok.tier.isna()] if tier is None else ok[ok.tier == tier]
        g_ep = ep[ep.tier.isna()] if tier is None else ep[ep.tier == tier]
        a_, b_ = (rstats(g_ep[(g_ep.date >= lo) & (g_ep.date < hi)]["r"])
                  for lo, hi in (("2021-01-01", "2025-01-01"), ("2025-01-01", "2027-01-01")))
        L.append(f"| {tier or 'unknown'} | {f(rstats(g_all['r']))} | {f(rstats(g_ep['r']))} | "
                 f"{a_['avgR']:+.2f} / {b_['avgR']:+.2f} |")
    L += ["", f"_Episode-first = first signal of a setup (>{EPISODE_GAP} bars since that "
          "ticker's previous signal)._", "",
          "## Portfolio — noise-aware", ""]
    if a.subsamples > 0:
        tables = robust_compare(frames, run, stat, bars, spy, spy_above, calendar, settings,
                                WINDOWS, baseline="base", n_subsamples=a.subsamples,
                                frac=a.frac, precomputed=res)
        L += robust_markdown(tables, a.subsamples, a.frac)

    for wn, lo, hi in WINDOWS:
        L += [f"## Portfolio — {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for k in frames:
            s = stat(res[k], spy, lo, hi)
            if s is None:
                L.append(f"| {k} | (insufficient) | | | | | | | |")
                continue
            sp0 = s["spy_ret"]
            L.append(f"| {k} | {s['ret']*100:+.0f}% | {s['dd']*100:.0f}% | {s['sharpe']:.2f} | "
                     f"{s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['pf']:.2f} | {s['mcl']} |")
        if sp0 is not None:
            L.append(f"| SPY | {sp0*100:+.0f}% | | | | | | | |")
        L.append("")

    L += ["## Which tiers each order actually traded (full period, closed trades)", "",
          "| variant | " + " | ".join(f"{t or 'unknown'}: n / avgR" for t in TIERS) + " |",
          "|---|" + "---|" * len(TIERS)]
    for k in frames:
        tr = res[k]["tr"]
        cells = []
        for t in TIERS:
            g = tr[tr.tier.isna()] if t is None else tr[tr.tier == t]
            s = rstats(g["r"]) if len(g) else {"n": 0}
            cells.append(f"{s['n']} / {s['avgR']:+.2f}" if s["n"] else "0")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias "
          "not corrected (today's universe). The live Decision Agent's own ranking judgment "
          "is not modelled — this tests the mechanical order only._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[pool] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
