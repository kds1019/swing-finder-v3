"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Tests the Super Trend indicator (ATR-based volatility bands, standard default
period=10/multiplier=3) as an alternative/additional current-trend confirmation to the
existing 20-session EMA200 slope gate (core.trend_context.EMA200_SLOPE_LOOKBACK_DAYS /
research/current_trend_gate_ab.py, already live). Motivated by reviewing an external paper
(IJSAT, 2026) that used Super Trend as a trend-confirmation feature — the paper itself is
low-rigor (single-stock test, undisclosed backtest dates, no reported transaction costs, and
the lead author has a dozen near-identical self-published papers swapping in different
indicator/ML buzzwords), so it is NOT trusted as evidence Super Trend works; the indicator
itself is a legitimate, independently well-known tool, so it's tested here on its own merits
against this system's own signal set and trade outcomes, per the usual backtest-first
convention.

Super Trend is a per-TICKER read (unlike the sector-trend test, which was per-sector) — it's
a direct alternative to the current-trend slope gate, so it's tested the same way: tag every
signal in today's live-gate cache with the STOCK's OWN Super Trend direction (up/down) as of
the signal date, then check (1) raw bucketed trade outcomes by that direction and (2) whether
gating on it helps the blended portfolio.

Super Trend computation (standard, e.g. TradingView's built-in indicator): ATR(10) via
Wilder's smoothing (RMA), basic bands at (H+L)/2 +- 3*ATR, "final" bands that only move in the
trend's favor (ratchet, same idea as this repo's own trailing-stop logic) unless price closes
through the opposite band, direction flips when price closes through the current band.
Recursive band logic can't be vectorized (each bar depends on the previous final band), so
it's computed as a plain per-ticker loop — fast enough at ~1300 bars x ~480 tickers.

Usage:  python -m research.supertrend_ab            (uses cached live-gate signals)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.trade_plan import TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE
from research.current_trend_gate_ab import stat

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
OUT = Path(__file__).resolve().parent / "supertrend_ab.md"

INITIAL = 100_000.0
MAX_POS_PCT, MAX_POS, SECTOR_CAP, SLIP_BPS, MAX_HOLD = 20.0, 6, 3, 5, 30
START = "2021-06-01"
TIER_ORDER = {"stabilising": 0, "forming": 1, None: 2, "still_falling": 3}

ST_PERIOD = 10
ST_MULTIPLIER = 3.0


def compute_supertrend_direction(df: pd.DataFrame, period: int = ST_PERIOD,
                                  multiplier: float = ST_MULTIPLIER) -> np.ndarray:
    """Standard Super Trend direction series (1=up, -1=down), aligned with df's rows.
    Sequential by construction (each bar's "final" band only ratchets in the trend's favor
    unless price closes through the opposite one) — not vectorizable, hence the plain loop."""
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    n = len(close)

    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    atr = np.empty(n)
    atr[0] = tr[0]
    alpha = 1.0 / period
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = np.empty(n)
    final_lower = np.empty(n)
    direction = np.empty(n, dtype=int)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    direction[0] = 1 if close[0] > final_upper[0] else -1

    for i in range(1, n):
        final_upper[i] = (basic_upper[i] if (basic_upper[i] < final_upper[i - 1]
                                              or close[i - 1] > final_upper[i - 1])
                          else final_upper[i - 1])
        final_lower[i] = (basic_lower[i] if (basic_lower[i] > final_lower[i - 1]
                                              or close[i - 1] < final_lower[i - 1])
                          else final_lower[i - 1])
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close[i] > final_upper[i] else -1

    return direction


def build_supertrend_table(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        if raw is None or len(raw) < ST_PERIOD + 5:
            continue
        direction = compute_supertrend_direction(raw)
        rows.append(pd.DataFrame({"ticker": t, "date": raw["Date"].values, "st_dir": direction}))
    return pd.concat(rows, ignore_index=True)


def run(sig: pd.DataFrame, bars, spy_above, calendar, settings) -> dict:
    """Same engine/costs/conventions as research/current_trend_gate_ab.py's run(), plus
    carrying supertrend_state from signal to open position to closed trade so results can be
    bucketed by it afterward. Mark-to-market forward-fills each held position's last known
    close on a day its own bar is missing (see current_trend_gate_ab.py's run() docstring)."""
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
                               "sector": p["sector"], "st_dir": p["st_dir"]})
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
                                "st_dir": s["st_dir"], "last_close": entry}
                sec[s["sector"]] = sec.get(s["sector"], 0) + 1
    return {"eq": pd.DataFrame(eq, columns=["date", "equity"]).set_index("date"),
            "tr": pd.DataFrame(closed)}


def bucket_stats(tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, g in tr.groupby("st_dir", dropna=False):
        r = g["r"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        label = {1: "up", -1: "down"}.get(state, state)
        rows.append({"st_dir": label, "trades": len(g), "win%": (r > 0).mean() * 100,
                     "avgR": r.mean(), "medianR": r.median(), "pf": pf})
    return pd.DataFrame(rows)


def main():
    settings = load_settings()
    if not CURRENT_TREND_SIG_CACHE.exists():
        sys.exit("research/data/current_trend_signals.pkl not found — run "
                  "`python -m research.current_trend_gate_ab` first to build it.")
    sig = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()
    print(f"[supertrend] {len(sig)} live-gate signals loaded", file=sys.stderr)

    tickers = sorted(sig.ticker.unique())
    st = build_supertrend_table(tickers)
    st["date"] = pd.to_datetime(st["date"]).dt.normalize()
    print(f"[supertrend] Super Trend computed for {st['ticker'].nunique()} tickers, "
          f"{len(st)} (ticker, date) rows", file=sys.stderr)

    sig = sig.merge(st, on=["ticker", "date"], how="left")
    unmatched = sig["st_dir"].isna().sum()
    print(f"[supertrend] {unmatched}/{len(sig)} signals had no exact (ticker, date) Super "
          f"Trend match", file=sys.stderr)

    bars = {t: pd.read_pickle(BARS_DIR / f"{t}.pkl").set_index("Date") for t in tickers}
    spy = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")["Close"]
    spy_above = spy > spy.rolling(200).mean()
    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})

    windows = [("full", "2021-01-01", "2027-01-01"), ("2021-2024", "2021-01-01", "2025-01-01"),
               ("2025-2026", "2025-01-01", "2027-01-01"), ("2022", "2022-01-01", "2023-01-01")]

    baseline = run(sig, bars, spy_above, calendar, settings)
    buckets = bucket_stats(baseline["tr"])

    L = ["# Super Trend (ATR-based, period=10/mult=3) as a current-trend confirmation "
         "(RESEARCH)", "",
         "Motivated by reviewing an external paper (IJSAT, 2026) that used Super Trend as a "
         "trend-confirmation input to a neural net — the paper itself is low-rigor (single "
         "stock, undisclosed backtest dates, no reported transaction costs, templated "
         "self-citations), so it's tested here on the indicator's own merits, not on the "
         "paper's authority. Super Trend computed the standard way (ATR(10) via Wilder's "
         "smoothing, bands at (H+L)/2 +- 3*ATR, direction flips when price closes through the "
         "current band) per ticker, tagged onto every signal in today's live gate "
         "(research/data/current_trend_signals.pkl, unmodified) by its own (ticker, date).",
         f"- signal set: today's live gate held exactly as-is, {len(sig)} signals, "
         f"{unmatched} unmatched", "",
         "## View 1 — realized trade outcomes, bucketed by the stock's OWN Super Trend "
         "direction at entry (one portfolio run, today's actual gate, no additional filter)",
         "",
         "| supertrend direction at entry | trades | win% | avgR | medianR | PF |",
         "|---|---|---|---|---|---|"]
    for _, row in buckets.iterrows():
        L.append(f"| {row['st_dir']} | {row['trades']:.0f} | {row['win%']:.0f}% | "
                 f"{row['avgR']:+.2f} | {row['medianR']:+.2f} | {row['pf']:.2f} |")
    L.append("")

    sig_variants = {"no gate (current)": sig, "require supertrend up": sig[sig.st_dir == 1]}
    res = {name: (baseline if name == "no gate (current)"
                  else run(s, bars, spy_above, calendar, settings))
           for name, s in sig_variants.items()}
    counts = {name: len(s) for name, s in sig_variants.items()}
    L.append("- signal counts per variant: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    L.append("")
    L.append("## View 2 — portfolio-level gate variant (does requiring Super Trend already "
             "flag 'up' help the blended portfolio?)")
    L.append("")

    for wn, lo, hi in windows:
        L += [f"### {wn}", "",
              "| variant | ret | maxDD | Sharpe | trades | win% | avgR | PF | maxConsecLoss |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp0 = None
        for name in sig_variants:
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
          "corrected._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[supertrend] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
