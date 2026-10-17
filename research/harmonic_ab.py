"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Do bullish harmonic patterns (Gartley, Bat — research/harmonic.py) help this pipeline?
Two questions, answered separately:

PART 1 — tag the screened signals. On the live candidate set (current screener gates + the
  weak-RR drop), do signals with an ACTIVE bullish pattern (price reached the pattern's
  potential reversal zone, not yet failed below X) do better than signals without one?
    1a  per-signal outcomes by tag (every signal simulated on its own, same exit rules)
    1b  portfolio run, closed trades bucketed by tag
    1c  portfolio variants (noise-aware, research/robust.py):
          base               live behavior exactly
          harmonic_priority  pool order = knife tier, THEN harmonic-active first, then depth
          require_harmonic   only harmonic-active signals are tradeable
    1d  stop vs X: is the live stop above the pattern's own invalidation level?

PART 2 — what the screener drops. Every bullish pattern in the frozen universe, on the bar
  price first reaches its reversal zone ("touch"), is run through the live screener
  (core.pullback_reversal.screener_gate — the single gate definition) and the weak-RR drop:
    2a  funnel: which gate rejected it on the touch bar, and whether the screener passed
        it on ANY bar while the pattern was active (caught later)
    2b  outcomes by funnel bucket — touch-bar entry, live trade plan, live exit rules — vs
        the whole live signal set as the reference
    2c  pool squeeze: on the days a harmonic stock did pass, was it outside the top
        CANDIDATE_POOL_SIZE after knife-tier ordering + pre-research sector cap?

Exit rules everywhere (per-signal and portfolio) are the live ones: entry = signal-bar close
(+5bps), core.trade_plan stop/target, hold the stop to +2R, then trail 1R below the running
high, target is a ceiling, 30-bar max hold. Portfolio engine/stat copied verbatim from
research/stop_floor_ab.py (6 positions, 3/sector, 20% max position, SPY>200SMA gate).
Per-signal stats count only signals on SPY>200SMA days, same as the portfolio.

Detector parameters are fixed in research/harmonic.py and were NOT tuned to these results.

Usage:  python -m research.harmonic_ab                    (fetch frozen tickers + build + report)
        python -m research.harmonic_ab --no-fetch         (use cached research/data/bars)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    MIN_BARS_FOR_SCREENER, classify_knife_risk, measure_pullback_reversal, measure_stabilization,
    screener_gate,
)
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan
from research import harmonic as H
from research.robust import frozen_universe, robust_compare, robust_markdown
from research.stop_floor_ab import fetch_bars

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "harmonic_rows.pkl"
OUT = Path(__file__).resolve().parent / "harmonic_ab.md"
CSV_DIR = Path(__file__).resolve().parent / "data"
START = "2021-06-01"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD, WINDOW = 20.0, 6, 3, 5, 30, 300
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}
POOL_SIZE = 30            # pipeline.CANDIDATE_POOL_SIZE (not imported: pipeline.py pulls in every agent)
EPISODE_GAP = 5           # a ticker's signal starts a new "episode" after this many signal-free bars
FWD_DAYS = 20

WINDOWS = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
           ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]


# ---------------------------------------------------------------- per-signal simulation
def simulate(high, low, close, i, entry, stop, target):
    """Live exit rules for one trade entered at bar i's close. Returns (r, reason) or (nan, None)
    if there are no bars after i."""
    frict = SLIP_BPS / 10000.0
    e = entry * (1 + frict)
    risk = e - stop
    if risk <= 0 or i + 1 >= len(close):
        return np.nan, None
    peak, active = e, False
    for held, j in enumerate(range(i + 1, len(close)), 1):
        eff = max(stop, peak - TRAIL_GIVEBACK_R * risk) if active else stop
        if low[j] <= eff:
            xp, why = eff, ("trail_stop" if eff > stop else "stop_hit")
        elif high[j] >= target:
            xp, why = target, "target_hit"
        elif held >= MAX_HOLD:
            xp, why = close[j], "expired"
        else:
            xp = None
        peak = max(peak, high[j])
        if not active and high[j] >= e + TRAIL_ACTIVATE_R * risk:
            active = True
        if xp is not None:
            return (xp * (1 - frict) - e) / risk, why
    return (close[-1] * (1 - frict) - e) / risk, "open"


# ---------------------------------------------------------------- build
def harmonic_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-bar harmonic state (first candidate per bar, preferring an in-zone one)."""
    s = H.harmonic_states(raw)
    if s.empty:
        return s
    s = s.sort_values(["idx", "in_zone"], ascending=[True, False])
    return s


def build_rows(settings, tickers: list[str], sectors: dict) -> pd.DataFrame:
    """One row per (ticker, bar) that is either a screener-prefilter bar or a harmonic
    touch bar, with the live gate verdict, knife tier, live trade plan, harmonic tags and
    the per-signal outcome."""
    rows = []
    t0 = time.time()
    for n, t in enumerate(tickers, 1):
        path = BARS_DIR / f"{t}.pkl"
        if not path.exists():
            continue
        raw = pd.read_pickle(path)
        if raw is None or len(raw) < MIN_BARS_FOR_SCREENER + 5:
            continue
        raw = raw.reset_index(drop=True)
        df = compute_indicators(raw.copy())
        high, low, close = (df[c].astype(float).to_numpy() for c in ("High", "Low", "Close"))
        hs = harmonic_table(raw)
        by_idx = {} if hs.empty else {i: g.iloc[0] for i, g in hs.groupby("idx")}
        touch_idx = {} if hs.empty else {
            int(r.idx): r for r in hs[hs.first_touch].itertuples()}
        # episode id per active candidate, so Part 2 can ask "did the screener catch it later?"
        cand_bars = {} if hs.empty else {
            k: set(g.idx.astype(int)) for k, g in hs.groupby(["c_confirm_idx", "pattern"])}

        ema200 = df["EMA200"]
        pvs = (df["Close"] / ema200 - 1.0) * 100.0
        upt = (ema200 / ema200.shift(126) - 1.0) * 100.0
        # Same superset prefilter as research/stop_floor_ab.py; the real gate runs below.
        net = ((upt > 0) & (pvs >= -25) & (df["Close"] <= df["EMA50"] * 1.02)).to_numpy()
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        cand = set(np.where(net)[0]) | set(touch_idx)
        for i in sorted(cand):
            if i < first or str(df["Date"].iloc[i].date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW)
            m = measure_pullback_reversal(prefix)
            gate = "insufficient_data" if m is None else screener_gate(m)
            if i not in touch_idx and gate is not None:
                continue            # plain screener reject with no harmonic event: not needed
            p = compute_trade_plan(prefix, settings)
            ok_plan = p is not None and p["stop"] < p["entry"]
            hrow = by_idx.get(i)
            row = {"date": df["Date"].iloc[i], "ticker": t, "sector": sectors.get(t, "Unknown"),
                   "i": i, "gate": gate or "pass",
                   "pvs": None if m is None else m["price_vs_ema200_pct"],
                   "tier": classify_knife_risk(measure_stabilization(prefix)),
                   "entry": p["entry"] if ok_plan else np.nan,
                   "stop": p["stop"] if ok_plan else np.nan,
                   "target": p["target"] if ok_plan else np.nan,
                   "weak_rr": bool(p["weak_rr"]) if ok_plan else True,
                   "harm_active": hrow is not None,
                   "harm_pattern": None if hrow is None else hrow["pattern"],
                   "harm_in_zone": False if hrow is None else bool(hrow["in_zone"]),
                   "harm_x": np.nan if hrow is None else hrow["x"],
                   "harm_touch": i in touch_idx}
            if i in touch_idx:
                tr = touch_idx[i]
                key = (tr.c_confirm_idx, tr.pattern)
                row["touch_pattern"] = tr.pattern
                row["touch_key"] = f"{t}|{tr.c_confirm_idx}|{tr.pattern}"
                row["touch_bars"] = sorted(cand_bars.get(key, {i}))
            if ok_plan:
                row["r"], row["exit"] = simulate(high, low, close, i, p["entry"], p["stop"], p["target"])
            else:
                row["r"], row["exit"] = np.nan, None
            j = min(i + FWD_DAYS, len(close) - 1)
            row["fwd20"] = (close[j] / close[i] - 1) * 100 if j > i else np.nan
            rows.append(row)
        if n % 50 == 0:
            print(f"[harm] scanned {n}/{len(tickers)} ({len(rows)} rows, {time.time()-t0:.0f}s)",
                  file=sys.stderr)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- portfolio (stop_floor_ab copy + tag/priority)
def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """research/stop_floor_ab.py::run(), plus: pool order is tier, then `_pri` (0 first),
    then depth; the harmonic tag is carried onto each closed trade."""
    d = sig.assign(_tr=sig.tier.map(TIER_ORDER).fillna(2))
    if "_pri" not in d:
        d["_pri"] = 0
    by_date = {dt: g.sort_values(["_tr", "_pri", "pvs"]) for dt, g in d.groupby("date")}
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
                               "sector": p["sector"], "weak_rr": False, "harm": p["harm"]})
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
                                "harm": bool(s["harm_active"]), "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def stat(res, spy, lo, hi):
    """research/stop_floor_ab.py::stat() verbatim."""
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


# ---------------------------------------------------------------- reporting helpers
def rstats(r: pd.Series) -> dict:
    r = r.dropna()
    if not len(r):
        return {"n": 0, "win": np.nan, "avgR": np.nan, "medR": np.nan, "pf": np.nan}
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
    return {"n": len(r), "win": (r > 0).mean() * 100, "avgR": r.mean(), "medR": r.median(), "pf": pf}


def table(groups: list[tuple[str, pd.DataFrame]], with_fwd: bool = False, split: bool = True) -> list[str]:
    hdr = "| bucket | n | win% | avgR | medianR | PF |"
    sep = "|---|---|---|---|---|---|"
    if with_fwd:
        hdr += " fwd20 mean | fwd20 >0 |"
        sep += "---|---|"
    if split:
        hdr += " avgR 2021-24 (n) | avgR 2025-26 (n) |"
        sep += "---|---|"
    L = [hdr, sep]
    for name, g in groups:
        s = rstats(g["r"])
        if s["n"] == 0:
            L.append(f"| {name} | 0 |" + " |" * (sep.count("|") - 3))
            continue
        line = (f"| {name} | {s['n']} | {s['win']:.0f}% | {s['avgR']:+.2f} | {s['medR']:+.2f} | "
                f"{s['pf']:.2f} |")
        if with_fwd:
            f = g["fwd20"].dropna()
            line += f" {f.mean():+.1f}% | {(f > 0).mean()*100:.0f}% |" if len(f) else " | |"
        if split:
            for lo, hi in (("2021-01-01", "2025-01-01"), ("2025-01-01", "2027-01-01")):
                gg = g[(g.date >= lo) & (g.date < hi)]
                ss = rstats(gg["r"])
                line += f" {ss['avgR']:+.2f} ({ss['n']}) |" if ss["n"] else " – (0) |"
        L.append(line)
    return L


def episode_first(sig: pd.DataFrame) -> pd.DataFrame:
    """First signal of each run of consecutive-ish signals per ticker (consecutive daily
    signals on one stock are one setup, not N independent trades)."""
    s = sig.sort_values(["ticker", "i"])
    new = s.groupby("ticker")["i"].diff().fillna(EPISODE_GAP + 1) > EPISODE_GAP
    return s[new.to_numpy()]


def pool_membership(live: pd.DataFrame, pre_cap: int) -> pd.Series:
    """For each live signal: would it have been inside the pipeline's candidate pool
    (knife tier, then depth; pre-research sector cap; head(POOL_SIZE))?"""
    inside = pd.Series(False, index=live.index)
    d = live.assign(_tr=live.tier.map(TIER_ORDER).fillna(2))
    for _, g in d.groupby("date"):
        g = g.sort_values(["_tr", "pvs"])
        g = g[g.groupby("sector").cumcount() < pre_cap]
        inside.loc[g.head(POOL_SIZE).index] = True
    return inside


# ---------------------------------------------------------------- main
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
        print(f"[harm] loaded {len(rows)} cached rows", file=sys.stderr)
    else:
        tickers = sorted(frozen["Ticker"])
        if not a.no_fetch:
            fetch_bars(settings, tickers, a.start)
        rows = build_rows(settings, tickers, sectors)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        rows.to_pickle(SIG_CACHE)
        print(f"[harm] built + cached {len(rows)} rows", file=sys.stderr)

    rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
    rows = rows[rows["ticker"].isin(sectors)].copy()
    rows["sector"] = rows["ticker"].map(sectors)
    tickers = sorted(rows.ticker.unique())
    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    spy_above.index = pd.to_datetime(spy_above.index).normalize()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})
    rows["spy_ok"] = rows["date"].map(spy_above).fillna(False).astype(bool)

    live_all = rows[(rows.gate == "pass") & (~rows.weak_rr)].copy()
    live = live_all[live_all.spy_ok]
    touches = rows[rows.harm_touch & rows.spy_ok].copy()

    L = ["# Bullish harmonic patterns (Gartley, Bat) — two-part test (RESEARCH)", "",
         f"- {len(tickers)} tickers (research/frozen_universe.csv), signals from {START}",
         f"- detector: research/harmonic.py — ZigZag {H.ZIGZAG_ATR_MULT}x ATR{H.ATR_PERIOD}, "
         f"pattern bands fixed in advance (see its docstring), active window "
         f"{H.ACTIVE_MAX_BARS} bars after the first touch, causal (no repainting)",
         f"- live signal set (screener pass + weak-RR drop, SPY>200SMA days): {len(live)} "
         f"signals, {live.harm_active.sum()} ({live.harm_active.mean():.1%}) with an active "
         f"bullish pattern",
         f"- harmonic touch events in the universe (SPY>200SMA days): {len(touches)} "
         f"({(touches.touch_pattern == 'gartley').sum()} Gartley, "
         f"{(touches.touch_pattern == 'bat').sum()} Bat)",
         "- per-signal R uses the live exit rules on each signal independently; consecutive "
         "daily signals on one stock overlap, so the 'episode-first' rows (first signal per "
         f"setup, >{EPISODE_GAP} bars apart) are the fairer count", ""]

    # ---------------- PART 1
    L += ["## Part 1 — harmonic tag on the screened signals", "",
          "### 1a. Per-signal outcomes by tag", "", "**All live signals**", ""]
    grp = lambda s: [("no pattern", s[~s.harm_active]), ("pattern active (any)", s[s.harm_active]),
                     ("  Gartley", s[s.harm_pattern == "gartley"]),
                     ("  Bat", s[s.harm_pattern == "bat"]),
                     ("  in zone today", s[s.harm_in_zone]),
                     ("  active, bouncing off zone", s[s.harm_active & ~s.harm_in_zone])]
    L += table(grp(live), with_fwd=True) + ["", "**Episode-first signals**", ""]
    ep = episode_first(live)
    L += table(grp(ep), with_fwd=True) + [""]
    L += ["**By knife tier (episode-first) — does the pattern add anything inside a tier?**", "",
          "| tier | no pattern: n / avgR / PF | pattern: n / avgR / PF |", "|---|---|---|"]
    for tier in ("stabilising", "forming", "still_falling"):
        g = ep[ep.tier == tier]
        a0, a1 = rstats(g[~g.harm_active]["r"]), rstats(g[g.harm_active]["r"])
        f = lambda s: f"{s['n']} / {s['avgR']:+.2f} / {s['pf']:.2f}" if s["n"] else "0"
        L.append(f"| {tier} | {f(a0)} | {f(a1)} |")
    L.append("")

    frames = {"base": live_all.assign(_pri=0),
              "harmonic_priority": live_all.assign(_pri=(~live_all.harm_active).astype(int)),
              "require_harmonic": live_all[live_all.harm_active].assign(_pri=0)}
    res = {k: run(v, bars, spy_above, calendar, settings) for k, v in frames.items()}

    L += ["### 1b. Portfolio (live engine), closed trades bucketed by tag", ""]
    tr = res["base"]["tr"]
    if len(tr):
        tr = tr.assign(date=pd.to_datetime(tr.exit_date))
        L += table([("no pattern", tr[~tr.harm]), ("pattern active", tr[tr.harm])]) + [""]

    L += ["### 1c. Portfolio variants", "",
          f"- signal counts: " + ", ".join(f"{k}={len(v)}" for k, v in frames.items()), ""]
    if a.subsamples > 0:
        tables = robust_compare(frames, run, stat, bars, spy, spy_above, calendar, settings,
                                WINDOWS, baseline="base", n_subsamples=a.subsamples,
                                frac=a.frac, precomputed=res)
        L += robust_markdown(tables, a.subsamples, a.frac)
    for wn, lo, hi in WINDOWS:
        L += [f"#### {wn}", "",
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

    hp = ep[ep.harm_active]
    if len(hp):
        above = hp[hp.stop > hp.harm_x]
        below = hp[hp.stop <= hp.harm_x]
        L += ["### 1d. Live stop vs the pattern's invalidation level X (episode-first, pattern active)", "",
              f"- live stop ABOVE X (inside the pattern's normal noise): {len(above)} "
              f"({len(above)/len(hp):.0%}); stop at/below X: {len(below)}", ""]
        L += table([("stop above X", above), ("stop at/below X", below)], split=False) + [""]

    # ---------------- PART 2
    L += ["## Part 2 — what the screener drops", "",
          "Every bullish pattern, on the bar price first reaches its reversal zone.", ""]
    live_keys = set(zip(live_all.ticker, live_all.i))
    touches["caught_ever"] = [any((t, b) in live_keys for b in bars_)
                              for t, bars_ in zip(touches.ticker, touches.touch_bars)]
    touches["bucket"] = np.where(
        touches.gate != "pass", "rejected: " + touches.gate,
        np.where(touches.weak_rr, "passed gates, dropped weak-RR", "passed (live signal)"))
    L += ["### 2a. Funnel on the touch bar", "",
          "| touch-bar verdict | events | share | caught on a later bar while active |",
          "|---|---|---|---|"]
    for b, g in touches.groupby("bucket"):
        L.append(f"| {b} | {len(g)} | {len(g)/len(touches):.0%} | "
                 f"{g.caught_ever.mean():.0%} |")
    never = touches[~touches.caught_ever]
    L += ["", f"- never a live signal at any point while the pattern was active: {len(never)} "
          f"of {len(touches)} ({len(never)/max(len(touches),1):.0%})", ""]

    L += ["### 2b. Outcomes (touch-bar entry, live plan + exits) vs the live signal set", "",
          "Rows with no valid live trade plan have no R (fwd20 still shown).", ""]
    groups = [("REFERENCE: all live signals (episode-first)", ep),
              ("all touch events", touches),
              ("touch passed (live signal)", touches[touches.bucket == "passed (live signal)"]),
              ("touch rejected, caught later", touches[(touches.gate != "pass") & touches.caught_ever]),
              ("never caught (any reason)", never)]
    for b in sorted(never.bucket.unique()):
        groups.append((f"  never caught — {b}", never[never.bucket == b]))
    for tier in ("stabilising", "forming", "still_falling"):
        groups.append((f"touch, tier {tier}", touches[touches.tier == tier]))
    L += table(groups, with_fwd=True) + [""]

    pre_cap = getattr(settings, "pre_research_sector_cap", 8)
    live_all["in_pool"] = pool_membership(live_all, pre_cap)
    per_day = live_all.groupby("date").size()
    hl = live_all[live_all.harm_active & live_all.spy_ok]
    L += ["### 2c. Candidate-pool squeeze", "",
          f"- days with more than {POOL_SIZE} live signals: {(per_day > POOL_SIZE).sum()} of "
          f"{len(per_day)} (frozen ~500-ticker universe; the live ~full universe binds far more often)",
          f"- harmonic-active live signals outside the pool: {(~hl.in_pool).sum()} of {len(hl)}",
          f"- all live signals outside the pool: {(~live_all.in_pool).sum()} of {len(live_all)}", ""]
    if (~hl.in_pool).any():
        L += table([("harmonic, in pool", hl[hl.in_pool]), ("harmonic, squeezed out", hl[~hl.in_pool])],
                   split=False) + [""]

    L += ["_Daily-bar sim: intraday whipsaw and real fills not modelled. Survivorship bias not "
          "corrected (today's universe). Detector parameters fixed before the run._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    keep = ["date", "ticker", "sector", "gate", "tier", "weak_rr", "pvs", "entry", "stop", "target",
            "r", "exit", "fwd20", "harm_active", "harm_pattern", "harm_in_zone", "harm_x"]
    rows[rows.gate == "pass"][keep].to_csv(CSV_DIR / "harmonic_live_signals.csv", index=False)
    touches[keep + ["touch_pattern", "bucket", "caught_ever"]].to_csv(
        CSV_DIR / "harmonic_touch_events.csv", index=False)
    print("\n".join(L))
    print(f"\n[harm] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
