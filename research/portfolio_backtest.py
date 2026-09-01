"""
Portfolio-level backtest of the whole screener + trailing exit — the capstone check.

The runs/ backtests are per-symbol with an independent account each; this one runs a
SINGLE account across the cached-bar universe with concurrent positions, a sector cap,
realistic position-size limits, and daily mark-to-market — i.e. "would trading what this
system actually signals, together, have compounded?"

Conventions match research/build_calibration_dataset.py / core.pick_tracking exactly so
the equity curve is a direct check on the per-trade calibration numbers:
  - entry  = the signal bar's close (+ slippage); resolution starts the next bar
  - exit   = core.trade_plan trailing stop (hold initial stop to +2R, then trail
             peak - 1R), fib target as a hard ceiling, 30-bar max hold
  - sizing = risk_per_trade_pct of equity / (entry - stop), capped at MAX_POSITION_PCT
             of equity and by available cash

Deviations from the live pipeline (disclosed): no Decision Agent (no historical
fundamentals/news), so ranking when more signals fire than there is capacity is by
pullback depth (deepest first — where the calibration found the edge), not the live
BounceOffLowPct sort; universe is the ~460 cached tickers (survivorship-biased to
today's screen), not a point-in-time universe.

Usage:  python -m research.portfolio_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import load_settings
from core.indicators import compute_indicators
from core.pullback_reversal import (
    MIN_BARS_FOR_SCREENER, PRICE_VS_EMA200_MAX_PCT, PRICE_VS_EMA200_MIN_PCT,
    EMA200_MIN_UPTREND_PCT, detect_pullback_reversal,
)
from core.trade_plan import (
    TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan,
)
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
OUT = Path(__file__).resolve().parent / "portfolio_backtest.md"

INITIAL_CAPITAL = 100_000.0
MAX_POSITION_PCT = 20.0     # no single position above this % of equity at entry
MAX_POSITIONS = 6
SECTOR_CAP = 3              # concurrent open positions per sector
SLIPPAGE_BPS = 5
MAX_HOLD_DAYS = 30
WINDOW_BARS = 300
START = "2021-06-01"       # ~when EMA200 is warm given 2020-01 history
REGIME_FILTER = "--no-regime" not in sys.argv   # only open when SPY > its 200-day SMA
                                                # (stands in for the live VIX<=20 gate)


def build_signals(tickers, sectors, settings):
    """All (date, ticker, plan) the screener would have fired, computed per-ticker with
    no portfolio state — each uses only bars up to its own date."""
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
        net = (upt >= EMA200_MIN_UPTREND_PCT) & pvs.between(PRICE_VS_EMA200_MIN_PCT - 2, PRICE_VS_EMA200_MAX_PCT + 2)
        first = max(MIN_BARS_FOR_SCREENER - 1, 300)
        for i in np.where(net.to_numpy())[0]:
            if i < first or i >= len(df) - 1:
                continue
            d = df["Date"].iloc[i]
            if str(d.date()) < START:
                continue
            prefix = df.iloc[: i + 1].tail(WINDOW_BARS)
            if not detect_pullback_reversal(prefix).get("detected"):
                continue
            plan = compute_trade_plan(prefix, settings)
            if plan is None or plan["stop"] >= plan["entry"]:
                continue
            sig.append({
                "date": d, "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "depth": float(pvs.iloc[i]), "weak_rr": bool(plan["weak_rr"]),
            })
        if n % 100 == 0:
            print(f"[bt] signals: {n} tickers, {len(sig)} so far", file=sys.stderr)
    return pd.DataFrame(sig)


def run() -> dict:
    settings = load_settings()
    uni = build_universe(settings)
    sectors = dict(zip(uni["Ticker"], uni["Sector"]))
    cached = sorted(p.stem for p in BARS_DIR.glob("*.pkl") if p.stem != "SPY")
    tickers = [t for t in cached if t in sectors]
    print(f"[bt] {len(tickers)} tickers with bars + sector", file=sys.stderr)

    bars = {}
    for t in tickers:
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        raw = raw.set_index("Date")
        bars[t] = raw
    spy_df = pd.read_pickle(BARS_DIR / "SPY.pkl").set_index("Date")
    spy = spy_df["Close"]
    spy_above_200 = (spy > spy.rolling(200).mean())

    signals = build_signals(tickers, sectors, settings)
    signals = signals.sort_values("date").reset_index(drop=True)
    sig_by_date = {d: g for d, g in signals.groupby("date")}
    print(f"[bt] {len(signals)} total signals, {signals['ticker'].nunique()} tickers", file=sys.stderr)

    calendar = sorted({d for t in tickers for d in bars[t].index if str(d.date()) >= START})
    frict = SLIPPAGE_BPS / 10000.0

    cash = INITIAL_CAPITAL
    positions: dict[str, dict] = {}
    equity_curve, closed = [], []

    for d in calendar:
        # --- manage / exit open positions ---
        for t in list(positions.keys()):
            p = positions[t]
            if d not in bars[t].index:
                continue
            bar = bars[t].loc[d]
            hi, lo, cl = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            risk = p["entry"] - p["stop"]
            eff_stop = p["stop"]
            if p["active"]:
                eff_stop = max(p["stop"], p["peak"] - TRAIL_GIVEBACK_R * risk)
            p["held"] += 1

            exit_px = exit_reason = None
            if lo <= eff_stop:
                exit_px, exit_reason = eff_stop, ("trail_stop" if eff_stop > p["stop"] else "stop_hit")
            elif hi >= p["target"]:
                exit_px, exit_reason = p["target"], "target_hit"
            elif p["held"] >= MAX_HOLD_DAYS:
                exit_px, exit_reason = cl, "expired"

            # update peak/activation AFTER the exit check (peak as of prior bar governs today's stop)
            p["peak"] = max(p["peak"], hi)
            if not p["active"] and hi >= p["entry"] + TRAIL_ACTIVATE_R * risk:
                p["active"] = True

            if exit_px is not None:
                proceeds = p["shares"] * exit_px * (1 - frict)
                cash += proceeds
                r = (exit_px - p["entry"]) / risk
                closed.append({"ticker": t, "entry_date": p["entry_date"], "exit_date": d,
                               "reason": exit_reason, "r_multiple": r,
                               "pnl": proceeds - p["shares"] * p["entry"], "held": p["held"],
                               "sector": p["sector"], "weak_rr": p["weak_rr"]})
                del positions[t]

        # --- mark to market ---
        mtm = cash + sum(
            pp["shares"] * float(bars[tt].loc[d, "Close"])
            for tt, pp in positions.items() if d in bars[tt].index
        )
        equity_curve.append((d, mtm))

        # --- new entries ---
        regime_ok = (not REGIME_FILTER) or bool(spy_above_200.get(d, False))
        if d in sig_by_date and regime_ok and len(positions) < MAX_POSITIONS:
            sec_count: dict[str, int] = {}
            for pp in positions.values():
                sec_count[pp["sector"]] = sec_count.get(pp["sector"], 0) + 1
            cand = sig_by_date[d].sort_values("depth")  # deepest pullback first
            for _, s in cand.iterrows():
                t = s["ticker"]
                if t in positions or len(positions) >= MAX_POSITIONS:
                    continue
                if sec_count.get(s["sector"], 0) >= SECTOR_CAP:
                    continue
                entry = s["entry"] * (1 + frict)
                risk_ps = entry - s["stop"]
                if risk_ps <= 0:
                    continue
                by_risk = (mtm * settings.risk_per_trade_pct / 100.0) / risk_ps
                by_cap = (mtm * MAX_POSITION_PCT / 100.0) / entry
                by_cash = cash / entry
                shares = int(min(by_risk, by_cap, by_cash))
                if shares <= 0:
                    continue
                cash -= shares * entry
                positions[t] = {"shares": shares, "entry": entry, "stop": s["stop"],
                                "target": s["target"], "peak": entry, "active": False,
                                "held": 0, "entry_date": d, "sector": s["sector"],
                                "weak_rr": s["weak_rr"]}
                sec_count[s["sector"]] = sec_count.get(s["sector"], 0) + 1

    # --- close anything still open at the last bar ---
    last_d = calendar[-1]
    for t, p in list(positions.items()):
        if last_d in bars[t].index:
            cl = float(bars[t].loc[last_d, "Close"])
            cash += p["shares"] * cl * (1 - frict)
            closed.append({"ticker": t, "entry_date": p["entry_date"], "exit_date": last_d,
                           "reason": "open_at_end", "r_multiple": (cl - p["entry"]) / (p["entry"] - p["stop"]),
                           "pnl": p["shares"] * (cl - p["entry"]), "held": p["held"],
                           "sector": p["sector"], "weak_rr": p["weak_rr"]})

    eq = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date")
    tr = pd.DataFrame(closed)
    spy_bt = spy.reindex(eq.index).ffill().bfill()

    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    total_ret = eq["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    cagr = (1 + total_ret) ** (1 / yrs) - 1
    dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
    rets = eq["equity"].pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else np.nan
    spy_ret = spy_bt.iloc[-1] / spy_bt.iloc[0] - 1
    spy_dd = (spy_bt / spy_bt.cummax() - 1).min()

    r = tr["r_multiple"]
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf

    lines = [
        "# Portfolio backtest — recalibrated screener + trailing exit", "",
        f"- universe: {len(tickers)} cached tickers (survivorship-biased), {START} .. {str(eq.index[-1].date())}",
        f"- {INITIAL_CAPITAL:,.0f} start, {settings.risk_per_trade_pct:g}% risk/trade, "
        f"max {MAX_POSITIONS} positions, {MAX_POSITION_PCT:g}% position cap, {SECTOR_CAP}/sector, {SLIPPAGE_BPS}bps slip",
        f"- exit: trail +{TRAIL_ACTIVATE_R:g}R activate / give {TRAIL_GIVEBACK_R:g}R, {MAX_HOLD_DAYS}-bar max hold",
        f"- regime filter (SPY > 200-SMA to open): {'ON' if REGIME_FILTER else 'OFF'}",
        "",
        "## Result", "",
        "| metric | strategy | SPY (same window) |",
        "|---|---|---|",
        f"| total return | {total_ret*100:+.1f}% | {spy_ret*100:+.1f}% |",
        f"| CAGR | {cagr*100:+.1f}% | {((1+spy_ret)**(1/yrs)-1)*100:+.1f}% |",
        f"| max drawdown | {dd*100:.1f}% | {spy_dd*100:.1f}% |",
        f"| Sharpe (daily, ann.) | {sharpe:.2f} | — |",
        "",
        "## Trades", "",
        f"- closed trades: {len(tr)}",
        f"- win rate (R>0): {(r>0).mean()*100:.1f}%",
        f"- avg R: {r.mean():+.3f}   median R: {r.median():+.2f}   profit factor: {pf:.2f}",
        f"- avg hold: {tr['held'].mean():.1f} bars",
        f"- exit reasons: {tr['reason'].value_counts().to_dict()}",
        f"- weak-RR share of trades taken: {tr['weak_rr'].mean()*100:.0f}%",
        "",
        "## By year", "",
        "| year | trades | win% | avg R | end equity |",
        "|---|---|---|---|---|",
    ]
    tr["yr"] = pd.to_datetime(tr["exit_date"]).dt.year
    for y, g in tr.groupby("yr"):
        ye = eq[eq.index.year == y]["equity"]
        lines.append(f"| {y} | {len(g)} | {(g.r_multiple>0).mean()*100:.0f}% | "
                     f"{g.r_multiple.mean():+.2f} | {ye.iloc[-1]:,.0f} |")
    lines += ["", "_Daily-bar sim: intraday whipsaw and real fills not modelled. "
              "Survivorship bias not corrected._"]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"cagr": cagr, "max_dd": dd, "pf": pf, "trades": len(tr)}


if __name__ == "__main__":
    run()
