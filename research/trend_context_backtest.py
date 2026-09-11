"""
Trend-context bucket backtest — Phase 2 of the trend-context layer (see core/trend_context.py
and docs/strategy.md). Answers the question the live pipeline is explicitly NOT trusted to
answer yet: does splitting "confirmed support" into trend_continuation vs. reversion_bounce,
and managing each differently, actually separate into a better and a worse bucket — before
either is wired into live ranking or position sizing.

Forked from research/portfolio_backtest.py rather than editing it in place, so that script's
numbers stay a stable, un-touched baseline to compare against. Same conventions (entry = signal
bar's close + slippage, resolution starts next bar, 30-bar max hold, cached bars in
research/data/bars/*.pkl, survivorship-biased to today's ~460-ticker cache) — see that file's
docstring for what's shared. What's different here:

  - Every signal is tagged with setup_type (core.trend_context.classify_setup_type), using the
    SAME stabilization_signal definition as the live pipeline (KnifeRiskTier == "stabilising").
    A signal that doesn't clear that bar (forming / still_falling knife risk) is tagged
    "unclassified" and simulated with the baseline's own (trailing, normal-size) behavior —
    the control group, not a third real bucket.
  - trend_continuation trades: unchanged from the baseline — the existing chandelier-style
    trail (+2R activate, trail peak-1R, never loosens), normal position size. This bucket is
    the one the whole trailing-exit rework (2026-08-31) was calibrated for.
  - reversion_bounce trades: trailing DISABLED — pure fixed stop/target (the pre-2026-08-31
    behavior), and sized at REVERSION_BOUNCE_SIZE_MULT of normal risk. This is a first-cut
    implementation of "fixed tight stop, quick in-and-out, sized smaller" using the *existing*,
    already-calibrated stop/target math (swing-low/EMA-anchored stop, Fibonacci-extension
    target) rather than inventing new untested thresholds — revisit the multiplier/targets
    once this backtest's numbers say whether the split is worth tuning further.
  - When more signals fire on a day than there's capacity for, trend_continuation candidates
    are prioritized over reversion_bounce ones (then deepest-pullback within a tier, matching
    the baseline's tie-break) — the deterministic version of "score the overlap higher."
  - Report breaks win rate / profit factor / avg R / avg hold out PER BUCKET, not just blended,
    plus the blended portfolio curve for comparison against portfolio_backtest.py's baseline.

Usage:  python -m research.trend_context_backtest
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
    EMA200_MIN_UPTREND_PCT, detect_pullback_reversal, measure_stabilization,
)
from core.trade_plan import (
    TRAIL_ACTIVATE_R, TRAIL_GIVEBACK_R, compute_trade_plan,
)
from core.trend_context import compute_trend_state, measure_swing_fib_retracement, classify_setup_type
from core.universe import build_universe

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"

INITIAL_CAPITAL = 100_000.0
MAX_POSITION_PCT = 20.0     # no single position above this % of equity at entry
MAX_POSITIONS = 6
SECTOR_CAP = 3              # concurrent open positions per sector
SLIPPAGE_BPS = 5
MAX_HOLD_DAYS = 30
WINDOW_BARS = 300
START = "2021-06-01"       # ~when EMA200 is warm given 2020-01 history
REGIME_FILTER = "--no-regime" not in sys.argv   # only open when SPY > its 200-day SMA

# Three candidate-selection-under-capacity modes, all sharing the same bucketed exit/sizing
# (trend_continuation trail vs. reversion_bounce fixed-stop-half-size) — only how the daily
# candidate list is ordered/gated differs. See docs/strategy.md.
#   "full"           (default) — hard priority: trend_continuation before reversion_bounce
#                     before unclassified. Gives trend_continuation a real sample (n=272 in
#                     the first run) but let it dominate every slot on days it had enough
#                     candidates — the likely cause of that run's -78.5% max drawdown
#                     (trend_continuation is only ~3% of raw signals, so hard-prioritizing it
#                     means whichever few names qualify, often correlated, get first claim on
#                     ALL 6 slots).
#   "depth_only"      (--depth-only-priority) — the ablation: reverts to the baseline's own
#                     tie-break (deepest pullback first), no setup_type priority at all.
#                     Isolated the drawdown to the reorder (DD back to ~baseline), but
#                     trend_continuation almost never wins a slot against the far more common
#                     other categories (n=2) — too few trades to read anything from.
#   "reserved_slots"  (--reserved-slots[=N], default N=2) — middle ground: up to N of the
#                     MAX_POSITIONS slots are reserved for trend_continuation candidates only
#                     (deepest first) each day; whatever's left fills depth-only from the
#                     WHOLE remaining pool (any setup_type, including leftover
#                     trend_continuation beyond the reserved N). Guarantees trend_continuation
#                     a real sample without letting it claim every slot the way "full" does.
DEFAULT_RESERVED_SLOTS = 2


def _parse_reserved_slots_arg() -> int | None:
    for arg in sys.argv:
        if arg == "--reserved-slots":
            return DEFAULT_RESERVED_SLOTS
        if arg.startswith("--reserved-slots="):
            return int(arg.split("=", 1)[1])
    return None


_reserved_arg = _parse_reserved_slots_arg()
if _reserved_arg is not None:
    PRIORITY_MODE = "reserved_slots"
    RESERVED_TREND_CONTINUATION_SLOTS = _reserved_arg
elif "--depth-only-priority" in sys.argv:
    PRIORITY_MODE = "depth_only"
    RESERVED_TREND_CONTINUATION_SLOTS = 0
else:
    PRIORITY_MODE = "full"
    RESERVED_TREND_CONTINUATION_SLOTS = 0

_OUT_NAMES = {
    "full": "trend_context_backtest.md",
    "depth_only": "trend_context_backtest_depth_only.md",
    "reserved_slots": "trend_context_backtest_reserved_slots.md",
}
OUT = Path(__file__).resolve().parent / _OUT_NAMES[PRIORITY_MODE]

# reversion_bounce trades are sized at this fraction of the normal risk_per_trade_pct sizing —
# a placeholder ratio (not itself calibrated) standing in for "sized smaller" until this
# backtest's numbers justify tuning it further. See module docstring.
REVERSION_BOUNCE_SIZE_MULT = 0.5

_SETUP_TYPE_PRIORITY = {"trend_continuation": 0, "reversion_bounce": 1}


def build_signals(tickers, sectors, settings):
    """All (date, ticker, plan, setup_type, trend context) the screener would have fired,
    computed per-ticker with no portfolio state — each uses only bars up to its own date.
    Same screener gate as research/portfolio_backtest.py (unmodified — see that module and
    core/pullback_reversal.py for why loosening it isn't needed to see reversion_bounce
    cases: a ticker can clear the EMA200-based gate while a plainer current SMA50/SMA200
    read still calls it a downtrend, e.g. the live CRUS case that motivated this)."""
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

            stab = measure_stabilization(prefix)
            trend = compute_trend_state(prefix)
            fib = measure_swing_fib_retracement(prefix)
            setup_type = classify_setup_type(
                trend.get("trend_state"), fib.get("in_fib_zone"),
                stab.get("knife_risk_tier") == "stabilising",
            ) or "unclassified"

            sig.append({
                "date": d, "ticker": t, "sector": sectors.get(t, "Unknown"),
                "entry": plan["entry"], "stop": plan["stop"], "target": plan["target"],
                "depth": float(pvs.iloc[i]), "weak_rr": bool(plan["weak_rr"]),
                "setup_type": setup_type, "trend_state": trend.get("trend_state"),
                "retracement_pct": fib.get("retracement_pct"),
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
    print(f"[bt] {len(signals)} total signals, {signals['ticker'].nunique()} tickers, "
          f"setup_type counts: {signals['setup_type'].value_counts().to_dict()}", file=sys.stderr)

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
            # reversion_bounce: trailing disabled — pure fixed stop/target (see module
            # docstring). Every other bucket keeps the baseline chandelier-style trail.
            if p["trailing_enabled"] and p["active"]:
                eff_stop = max(p["stop"], p["peak"] - TRAIL_GIVEBACK_R * risk)
            p["held"] += 1

            exit_px = exit_reason = None
            if lo <= eff_stop:
                exit_px, exit_reason = eff_stop, ("trail_stop" if eff_stop > p["stop"] else "stop_hit")
            elif hi >= p["target"]:
                exit_px, exit_reason = p["target"], "target_hit"
            elif p["held"] >= MAX_HOLD_DAYS:
                exit_px, exit_reason = cl, "expired"

            if p["trailing_enabled"]:
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
                               "sector": p["sector"], "weak_rr": p["weak_rr"],
                               "setup_type": p["setup_type"]})
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

            def try_open(s) -> bool:
                """Attempts to open one position from candidate row `s`; returns whether it
                did. Same sizing/exit rule for every mode — only candidate ORDER/GATING
                differs above this function."""
                nonlocal cash
                t = s["ticker"]
                if t in positions or len(positions) >= MAX_POSITIONS:
                    return False
                if sec_count.get(s["sector"], 0) >= SECTOR_CAP:
                    return False
                entry = s["entry"] * (1 + frict)
                risk_ps = entry - s["stop"]
                if risk_ps <= 0:
                    return False
                is_reversion = s["setup_type"] == "reversion_bounce"
                size_mult = REVERSION_BOUNCE_SIZE_MULT if is_reversion else 1.0
                by_risk = (mtm * settings.risk_per_trade_pct * size_mult / 100.0) / risk_ps
                by_cap = (mtm * MAX_POSITION_PCT * size_mult / 100.0) / entry
                by_cash = cash / entry
                shares = int(min(by_risk, by_cap, by_cash))
                if shares <= 0:
                    return False
                cash -= shares * entry
                positions[t] = {"shares": shares, "entry": entry, "stop": s["stop"],
                                "target": s["target"], "peak": entry, "active": False,
                                "held": 0, "entry_date": d, "sector": s["sector"],
                                "weak_rr": s["weak_rr"], "setup_type": s["setup_type"],
                                "trailing_enabled": not is_reversion}
                sec_count[s["sector"]] = sec_count.get(s["sector"], 0) + 1
                return True

            day_df = sig_by_date[d]
            if PRIORITY_MODE == "reserved_slots":
                # Phase A: up to RESERVED_TREND_CONTINUATION_SLOTS reserved for
                # trend_continuation only (deepest first).
                tc_cand = day_df[day_df["setup_type"] == "trend_continuation"].sort_values("depth")
                reserved_opened = 0
                for _, s in tc_cand.iterrows():
                    if reserved_opened >= RESERVED_TREND_CONTINUATION_SLOTS:
                        break
                    if try_open(s):
                        reserved_opened += 1
                # Phase B: whatever's left, depth-only, from the WHOLE remaining pool (any
                # setup_type, including leftover trend_continuation past the reserved cap).
                rest_cand = day_df[~day_df["ticker"].isin(positions.keys())].sort_values("depth")
                for _, s in rest_cand.iterrows():
                    try_open(s)
            elif PRIORITY_MODE == "depth_only":
                for _, s in day_df.sort_values("depth").iterrows():
                    try_open(s)
            else:  # "full" — hard priority: trend_continuation > reversion_bounce > unclassified
                cand = day_df.copy()
                cand["_prio"] = cand["setup_type"].map(_SETUP_TYPE_PRIORITY).fillna(2)
                for _, s in cand.sort_values(["_prio", "depth"]).iterrows():
                    try_open(s)

    # --- close anything still open at the last bar ---
    last_d = calendar[-1]
    for t, p in list(positions.items()):
        if last_d in bars[t].index:
            cl = float(bars[t].loc[last_d, "Close"])
            cash += p["shares"] * cl * (1 - frict)
            closed.append({"ticker": t, "entry_date": p["entry_date"], "exit_date": last_d,
                           "reason": "open_at_end", "r_multiple": (cl - p["entry"]) / (p["entry"] - p["stop"]),
                           "pnl": p["shares"] * (cl - p["entry"]), "held": p["held"],
                           "sector": p["sector"], "weak_rr": p["weak_rr"],
                           "setup_type": p["setup_type"]})

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

    def bucket_stats(g: pd.DataFrame) -> dict:
        r = g["r_multiple"]
        pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else np.inf
        return {
            "trades": len(g),
            "win_rate_pct": round((r > 0).mean() * 100, 1) if len(g) else None,
            "avg_r": round(r.mean(), 3) if len(g) else None,
            "median_r": round(r.median(), 2) if len(g) else None,
            "profit_factor": round(pf, 2) if len(g) else None,
            "avg_hold_bars": round(g["held"].mean(), 1) if len(g) else None,
        }

    _priority_descs = {
        "full": "HARD PRIORITY: trend_continuation > reversion_bounce > unclassified, "
                "deepest pullback first within a tier",
        "depth_only": "DEPTH-ONLY (ablation: matches baseline's own tie-break, setup_type "
                      "priority OFF — isolates the exit/sizing split from the reorder effect)",
        "reserved_slots": f"RESERVED SLOTS: up to {RESERVED_TREND_CONTINUATION_SLOTS} of "
                           f"{MAX_POSITIONS} slots reserved for trend_continuation (deepest "
                           f"first), remainder filled depth-only from the whole pool",
    }
    _title_suffixes = {
        "full": "", "depth_only": " (depth-only-priority ablation)",
        "reserved_slots": f" (reserved-slots={RESERVED_TREND_CONTINUATION_SLOTS} variant)",
    }
    _intro_suffixes = {
        "full": "",
        "depth_only": " This run reverts candidate-selection priority to depth-only, to "
                      "isolate whether the setup_type reorder (not the exit/sizing split) "
                      "drove the full-priority run's drawdown.",
        "reserved_slots": f" This run reserves {RESERVED_TREND_CONTINUATION_SLOTS} of "
                           f"{MAX_POSITIONS} slots/day for trend_continuation candidates only, "
                           f"then fills the rest depth-only from the whole pool — a middle "
                           f"ground between the full-priority run (real trend_continuation "
                           f"sample, but it could claim every slot) and the depth-only ablation "
                           f"(sane drawdown, but trend_continuation almost never got a slot).",
    }
    priority_desc = _priority_descs[PRIORITY_MODE]
    lines = [
        "# Trend-context bucket backtest — trend_continuation vs. reversion_bounce"
        + _title_suffixes[PRIORITY_MODE], "",
        "Phase 2 of core/trend_context.py's trend-context layer — see docs/strategy.md. "
        "Compare against research/portfolio_backtest.md (the un-bucketed baseline this was "
        "forked from) for the blended-portfolio effect of the split."
        + _intro_suffixes[PRIORITY_MODE],
        "",
        f"- universe: {len(tickers)} cached tickers (survivorship-biased), {START} .. {str(eq.index[-1].date())}",
        f"- {INITIAL_CAPITAL:,.0f} start, {settings.risk_per_trade_pct:g}% risk/trade "
        f"(reversion_bounce sized at {REVERSION_BOUNCE_SIZE_MULT:g}x that), "
        f"max {MAX_POSITIONS} positions, {MAX_POSITION_PCT:g}% position cap, {SECTOR_CAP}/sector, {SLIPPAGE_BPS}bps slip",
        f"- trend_continuation exit: trail +{TRAIL_ACTIVATE_R:g}R activate / give {TRAIL_GIVEBACK_R:g}R "
        f"(unchanged baseline); reversion_bounce exit: fixed stop/target, trailing OFF; "
        f"both capped at {MAX_HOLD_DAYS}-bar max hold",
        f"- regime filter (SPY > 200-SMA to open): {'ON' if REGIME_FILTER else 'OFF'}",
        f"- candidate priority when signals compete for capacity: {priority_desc}",
        "",
        "## Blended portfolio result", "",
        "| metric | strategy | SPY (same window) |",
        "|---|---|---|",
        f"| total return | {total_ret*100:+.1f}% | {spy_ret*100:+.1f}% |",
        f"| CAGR | {cagr*100:+.1f}% | {((1+spy_ret)**(1/yrs)-1)*100:+.1f}% |",
        f"| max drawdown | {dd*100:.1f}% | {spy_dd*100:.1f}% |",
        f"| Sharpe (daily, ann.) | {sharpe:.2f} | — |",
        "",
        "## Per-bucket trade stats (the actual A/B)", "",
        "| setup_type | trades | win% | avg R | median R | profit factor | avg hold (bars) |",
        "|---|---|---|---|---|---|---|",
    ]
    if not tr.empty:
        for setup_type in ["trend_continuation", "reversion_bounce", "unclassified"]:
            g = tr[tr["setup_type"] == setup_type]
            bs = bucket_stats(g)
            lines.append(
                f"| {setup_type} | {bs['trades']} | "
                f"{bs['win_rate_pct'] if bs['win_rate_pct'] is not None else '—'} | "
                f"{bs['avg_r'] if bs['avg_r'] is not None else '—'} | "
                f"{bs['median_r'] if bs['median_r'] is not None else '—'} | "
                f"{bs['profit_factor'] if bs['profit_factor'] is not None else '—'} | "
                f"{bs['avg_hold_bars'] if bs['avg_hold_bars'] is not None else '—'} |"
            )
        bs_all = bucket_stats(tr)
        lines.append(f"| **all (blended)** | {bs_all['trades']} | {bs_all['win_rate_pct']} | "
                     f"{bs_all['avg_r']} | {bs_all['median_r']} | {bs_all['profit_factor']} | "
                     f"{bs_all['avg_hold_bars']} |")
    lines += [
        "",
        "## By year and bucket", "",
        "| year | setup_type | trades | win% | avg R |",
        "|---|---|---|---|---|",
    ]
    if not tr.empty:
        tr["yr"] = pd.to_datetime(tr["exit_date"]).dt.year
        for (y, setup_type), g in tr.groupby(["yr", "setup_type"]):
            lines.append(f"| {y} | {setup_type} | {len(g)} | "
                         f"{(g.r_multiple>0).mean()*100:.0f}% | {g.r_multiple.mean():+.2f} |")
    lines += ["", "_Daily-bar sim: intraday whipsaw and real fills not modelled. "
              "Survivorship bias not corrected. reversion_bounce sizing/exit constants "
              "(REVERSION_BOUNCE_SIZE_MULT) are a first cut, not independently calibrated — "
              "revisit once this backtest's per-bucket numbers are in."]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return {"cagr": cagr, "max_dd": dd, "pf_all": bucket_stats(tr)["profit_factor"] if not tr.empty else None,
            "trades": len(tr)}


if __name__ == "__main__":
    run()
