"""
Backtest run script for core.pullback_reversal + core.trade_plan against real
historical Alpaca bars. Imports the project's actual core/*.py modules directly
(not re-implemented) so this tests the live scanner's real behavior.

See notes.md for the confirmed strategy interpretation, assumptions, and
disclosed limitations. See config.json for run parameters.

Prerequisite: raw/bars_<SYMBOL>.json for each symbol in config.json, fetched via
the `alpaca data bars` commands in notes.md's "How to actually run this" section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
REPO_ROOT = RUN_DIR.parents[1]  # runs/<run>/run.py -> repo root
sys.path.insert(0, str(REPO_ROOT))

from core.indicators import compute_indicators  # noqa: E402
from core.pullback_reversal import detect_pullback_reversal, MIN_BARS_FOR_SCREENER  # noqa: E402
from core.trade_plan import compute_trade_plan, resolve_trade_plan_outcome  # noqa: E402

CONFIG = json.loads((RUN_DIR / "config.json").read_text())
RAW_DIR = RUN_DIR / "raw"


class _Settings:
    """Matches compute_trade_plan's real settings.min_risk_reward contract without
    depending on config/settings.py's other fields (FMP/Alpaca keys, etc.) this
    script has no reason to need."""
    min_risk_reward = CONFIG["min_risk_reward"]


def load_bars(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"bars_{symbol}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Fetch it first — see notes.md's "
            "'How to actually run this' section for the exact `alpaca data bars` command."
        )
    raw = json.loads(path.read_text())
    bars = raw["bars"] if isinstance(raw, dict) and "bars" in raw else raw
    df = pd.DataFrame(bars)
    df = df.rename(columns={"t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date").reset_index(drop=True)
    return df


def data_fingerprint(raw_df: pd.DataFrame) -> dict:
    return {
        "provider": "alpaca",
        "access_method": "alpaca_cli",
        "feed": CONFIG["feed"],
        "adjustment": CONFIG["adjustment"],
        "timeframe": CONFIG["timeframe"],
        "extended_hours": False,
        "total_bars_fetched": len(raw_df),
        "bars_after_filter": len(raw_df),
        "first_bar_ts": str(raw_df["Date"].iloc[0]) if len(raw_df) else None,
        "last_bar_ts": str(raw_df["Date"].iloc[-1]) if len(raw_df) else None,
        "close_sum": float(raw_df["Close"].sum()) if len(raw_df) else 0.0,
        "volume_sum": float(raw_df["Volume"].sum()) if len(raw_df) else 0.0,
    }


def run_symbol(symbol: str) -> dict:
    """Walks the symbol's full bar history once, evaluating detect_pullback_reversal
    at every eligible bar. On a detected signal, computes the real trade plan, fills
    at next_open (bar T+1's open, per notes.md's fill-model decision), and resolves
    the outcome via the same resolve_trade_plan_outcome() the live pipeline uses to
    score its own past picks. Skips ahead to the resolution bar before resuming
    scanning -- only one open position per symbol at a time, matching how the live
    system would actually behave (you wouldn't take a second entry mid-trade)."""
    raw_df = load_bars(symbol)
    fingerprint = data_fingerprint(raw_df)
    df = compute_indicators(raw_df.copy())
    settings = _Settings()

    friction_pct = CONFIG["slippage_bps"] / 10000.0
    equity = float(CONFIG["initial_cash"])
    equity_curve: list[tuple] = []
    trades: list[dict] = []
    round_trips: list[dict] = []

    n = len(df)
    i = MIN_BARS_FOR_SCREENER - 1  # first index with exactly MIN_BARS_FOR_SCREENER rows available
    warmup_start_price = float(df["Open"].iloc[i + 1]) if i + 1 < n else None

    while i < n - 1:  # need at least one more bar for the next_open fill
        row_date = df["Date"].iloc[i]
        window_df = df.iloc[: i + 1]

        signal = detect_pullback_reversal(window_df)
        if not signal.get("detected"):
            equity_curve.append((row_date, equity))
            i += 1
            continue

        plan = compute_trade_plan(window_df, settings)
        if plan is None:
            equity_curve.append((row_date, equity))
            i += 1
            continue

        fill_bar = df.iloc[i + 1]
        entry_fill = float(fill_bar["Open"]) * (1 + friction_pct)
        stop, target = plan["stop"], plan["target"]

        if entry_fill <= stop:
            # Gapped through the stop before the fill could even happen.
            equity_curve.append((row_date, equity))
            i += 1
            continue

        risk_amount = equity * (CONFIG["risk_per_trade_pct"] / 100.0)
        shares = int(risk_amount // abs(entry_fill - stop))
        if shares <= 0:
            equity_curve.append((row_date, equity))
            i += 1
            continue

        after_bars = df.iloc[i + 2:].reset_index(drop=True)
        outcome, outcome_price, outcome_date, bars_to_resolution = resolve_trade_plan_outcome(
            after_bars, stop, target, CONFIG["max_hold_days"]
        )

        if outcome is None:
            # Not enough future bars in the fetched range to resolve this signal --
            # exclude it from round-trip stats entirely rather than guess.
            equity_curve.append((row_date, equity))
            break

        exit_fill = outcome_price * (1 - friction_pct)
        pnl = (exit_fill - entry_fill) * shares
        equity += pnl

        trades.append({"symbol": symbol, "side": "buy", "date": str(fill_bar["Date"]),
                        "price": round(entry_fill, 4), "shares": shares})
        trades.append({"symbol": symbol, "side": "sell", "date": str(outcome_date),
                        "price": round(exit_fill, 4), "shares": shares})
        round_trips.append({
            "symbol": symbol,
            "entry_date": str(fill_bar["Date"]), "entry_price": round(entry_fill, 4),
            "exit_date": str(outcome_date), "exit_price": round(exit_fill, 4),
            "shares": shares, "stop": round(stop, 4), "target": round(target, 4),
            "outcome": outcome, "bars_to_resolution": bars_to_resolution,
            "pnl": round(pnl, 2),
            "r_multiple": round(pnl / risk_amount, 3) if risk_amount else None,
            "ema200_uptrend_pct": signal["ema200_uptrend_pct"],
            "price_vs_ema200_pct": signal["price_vs_ema200_pct"],
            "consolidation_range_pct": signal["consolidation_range_pct"],
            "bounce_off_low_pct": signal["bounce_off_low_pct"],
            "price_vs_poc_pct": signal.get("price_vs_poc_pct"),
            "rr_ratio_planned": plan["rr_ratio"],
            "weak_rr": plan["weak_rr"],
            "stop_distance_sanity_flag": plan["stop_distance_sanity_flag"],
        })

        # Equity curve: step function (pre-trade equity held flat through the trade,
        # then step to post-trade equity at the exit bar) -- see notes.md's disclosed
        # limitation on mark-to-market granularity.
        exit_i = min(i + 2 + bars_to_resolution - 1, n - 1)
        for j in range(i, exit_i):
            equity_curve.append((df["Date"].iloc[j], equity - pnl))
        equity_curve.append((df["Date"].iloc[exit_i], equity))

        i = exit_i + 1  # resume scanning only after this trade has fully resolved

    return {
        "symbol": symbol,
        "fingerprint": fingerprint,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "trades": trades,
        "round_trips": round_trips,
        "raw_df": df,
        "warmup_start_price": warmup_start_price,
    }


def benchmark_buy_and_hold(df: pd.DataFrame, start_price: float, cash: float) -> list[tuple]:
    start_i = MIN_BARS_FOR_SCREENER  # same warmup boundary as the strategy
    shares = cash / start_price
    return [
        (row["Date"], shares * float(row["Close"]))
        for _, row in df.iloc[start_i:].iterrows()
    ]


def sharpe(equity_series: pd.Series) -> float:
    daily = equity_series.pct_change().dropna()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return 0.0
    return float(daily.mean() / daily.std(ddof=1) * (252 ** 0.5))


def max_drawdown(equity_series: pd.Series) -> float:
    if equity_series.empty:
        return 0.0
    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1
    return float(drawdown.min())


DISCLOSURE = """> **Important disclosure**
> This backtest is a hypothetical historical simulation and does not represent actual
> trading performance. Backtested results do not guarantee future results. Results
> depend on market-data quality, data feed selection, corporate-action handling, fees,
> slippage, liquidity, taxes, execution assumptions, and implementation details. This
> material is for research and educational purposes only and is not investment
> advice, a recommendation, an offer, or a solicitation to buy or sell securities,
> options, cryptocurrencies, or any other financial product. All investments involve
> risk and may lose value. Review Alpaca's disclosures and agreements at
> [alpaca.markets/disclosures](https://alpaca.markets/disclosures)."""


def _pct(x) -> str:
    return f"{x:.2%}" if x is not None else "n/a"


def write_report(summary: dict, rt_df: pd.DataFrame) -> str:
    lines = ["# Backtest report — core.pullback_reversal + core.trade_plan", ""]

    lines.append("## Performance vs Benchmarks")
    lines.append("")
    lines.append("| Symbol | Total Return | Strategy Sharpe | Strategy Max DD | Benchmark Return | Benchmark Sharpe | Benchmark Max DD | Round Trips |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for symbol, s in summary["per_symbol"].items():
        lines.append(
            f"| {symbol} | {_pct(s['total_return'])} | {s['sharpe']} | {_pct(s['max_drawdown'])} | "
            f"{_pct(s['benchmark_total_return'])} | {s['benchmark_sharpe']} | {_pct(s['benchmark_max_drawdown'])} | "
            f"{s['num_round_trips']} |"
        )
    lines.append("")

    m = summary["metrics"]
    lines.append("## Aggregate pattern metrics (all symbols combined)")
    lines.append("")
    lines.append(f"- Round trips: {m['num_round_trips']}")
    lines.append(f"- Hit rate: {m['hit_rate']}")
    lines.append(f"- Profit factor: {m['profit_factor']}")
    lines.append(f"- Average R-multiple: {m['avg_r_multiple']}")
    lines.append("")

    lines.append("## Strategy configuration")
    lines.append("")
    lines.append(f"- Symbols: {', '.join(summary['symbols'])}")
    lines.append(f"- Period: {summary['start']} to {summary['end']}")
    lines.append(f"- Timeframe: {summary['timeframe']}")
    lines.append(f"- Fill model: {summary['fill_model']}, {summary['slippage_bps']} bps slippage")
    lines.append(f"- Initial cash per symbol: ${summary['initial_cash']:,}")
    lines.append("")

    if len(rt_df):
        first = rt_df.iloc[0]
        last = rt_df.iloc[-1]
        lines.append("## First and last trade")
        lines.append("")
        lines.append(f"- First: {first['symbol']} entered {first['entry_date']} @ {first['entry_price']}, "
                      f"exited {first['exit_date']} @ {first['exit_price']} ({first['outcome']})")
        lines.append(f"- Last: {last['symbol']} entered {last['entry_date']} @ {last['entry_price']}, "
                      f"exited {last['exit_date']} @ {last['exit_price']} ({last['outcome']})")
        lines.append("")

    lines.append("## Assumptions")
    lines.append("")
    for a in summary["assumptions"]:
        lines.append(f"- {a}")
    lines.append("")

    if summary["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for w in summary["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Data fingerprint")
    lines.append("")
    for symbol, s in summary["per_symbol"].items():
        fp = s["data_fingerprint"]
        lines.append(f"- {symbol}: {fp['total_bars_fetched']} bars, {fp['first_bar_ts']} to {fp['last_bar_ts']}, "
                      f"feed={fp['feed']}, adjustment={fp['adjustment']}, close_sum={fp['close_sum']:.2f}")
    lines.append("")

    lines.append("## Caveats (see notes.md for full detail)")
    lines.append("")
    lines.append("- Universe/survivorship bias: symbols are today's live scan output replayed over history, "
                  "not a point-in-time historical universe reconstruction.")
    lines.append("- Equity curve steps at trade open/close, not full daily mark-to-market of open positions — "
                  "Sharpe/max-drawdown are approximate. Hit rate/profit-factor/avg-R above are not affected.")
    lines.append("- Trading-activity fees (SEC/FINRA/etc.) excluded — only slippage modeled, $0 commissions.")
    lines.append("")

    lines.append(DISCLOSURE)
    lines.append("")

    return "\n".join(lines)


def main():
    per_symbol_results = {s: run_symbol(s) for s in CONFIG["symbols"]}

    all_round_trips = []
    all_trades = []
    per_symbol_summary = {}
    warnings = []

    for symbol, result in per_symbol_results.items():
        all_round_trips.extend(result["round_trips"])
        all_trades.extend(result["trades"])

        eq_df = pd.DataFrame(result["equity_curve"], columns=["Date", "equity"]).drop_duplicates("Date")
        eq_df.to_csv(RUN_DIR / f"equity_{symbol}.csv", index=False)

        if result["warmup_start_price"]:
            bench_series = benchmark_buy_and_hold(
                result["raw_df"], result["warmup_start_price"], CONFIG["initial_cash"]
            )
            bench_df = pd.DataFrame(bench_series, columns=["Date", "equity"]).drop_duplicates("Date")
            bench_df.to_csv(RUN_DIR / f"benchmark_equity_{symbol}.csv", index=False)
        else:
            bench_df = pd.DataFrame(columns=["Date", "equity"])

        total_return = result["final_equity"] / CONFIG["initial_cash"] - 1
        bench_total_return = (
            bench_df["equity"].iloc[-1] / CONFIG["initial_cash"] - 1 if len(bench_df) else None
        )

        per_symbol_summary[symbol] = {
            "final_equity": round(result["final_equity"], 2),
            "total_return": round(total_return, 4),
            "benchmark_total_return": round(bench_total_return, 4) if bench_total_return is not None else None,
            "sharpe": round(sharpe(eq_df["equity"]), 3) if len(eq_df) else None,
            "benchmark_sharpe": round(sharpe(bench_df["equity"]), 3) if len(bench_df) else None,
            "max_drawdown": round(max_drawdown(eq_df["equity"]), 4) if len(eq_df) else None,
            "benchmark_max_drawdown": round(max_drawdown(bench_df["equity"]), 4) if len(bench_df) else None,
            "num_round_trips": len(result["round_trips"]),
            "data_fingerprint": result["fingerprint"],
        }

        if len(result["round_trips"]) == 0:
            warnings.append(
                f"{symbol}: zero completed round trips in {CONFIG['start']}..{CONFIG['end']} -- "
                "either the signal never fired, or every occurrence's 30-bar resolution "
                "window ran past the end of the fetched data."
            )

    rt_df = pd.DataFrame(all_round_trips)
    rt_df.to_csv(RUN_DIR / "round_trips.csv", index=False)
    pd.DataFrame(all_trades).to_csv(RUN_DIR / "trades.csv", index=False)

    if len(rt_df):
        hit_rate = float((rt_df["pnl"] > 0).mean())
        winning = rt_df.loc[rt_df["pnl"] > 0, "pnl"].sum()
        losing = rt_df.loc[rt_df["pnl"] < 0, "pnl"].sum()
        profit_factor = (
            float(winning / abs(losing)) if losing < 0 else (float("inf") if winning > 0 else 0.0)
        )
        avg_r = float(rt_df["r_multiple"].mean())
    else:
        hit_rate = profit_factor = avg_r = None

    summary = {
        "strategy_name": "core.pullback_reversal + core.trade_plan (SwingFinder live scanner)",
        "start": CONFIG["start"], "end": CONFIG["end"],
        "symbols": CONFIG["symbols"], "timeframe": CONFIG["timeframe"],
        "initial_cash": CONFIG["initial_cash"],
        "fill_model": CONFIG["fill_model"], "slippage_bps": CONFIG["slippage_bps"],
        "metrics": {
            "num_round_trips": len(rt_df),
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "profit_factor": profit_factor,
            "avg_r_multiple": round(avg_r, 3) if avg_r is not None else None,
        },
        "per_symbol": per_symbol_summary,
        "assumptions": [
            "next_open fill model (signal on bar T close, fill at bar T+1 open) -- no look-ahead bias",
            "5 bps slippage on entry and exit fills; $0 commissions",
            "4% of current equity risked per trade, independent $100,000 account per symbol",
            "one open position per symbol at a time",
            "symbols are today's live scan universe replayed over history (survivorship bias, disclosed in notes.md)",
        ],
        "warnings": warnings,
        "artifacts": {
            "trades": "trades.csv", "round_trips": "round_trips.csv",
            "equity": [f"equity_{s}.csv" for s in CONFIG["symbols"]],
            "benchmark_equity": [f"benchmark_equity_{s}.csv" for s in CONFIG["symbols"]],
            "report": "report.md",
        },
    }
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (RUN_DIR / "report.md").write_text(write_report(summary, rt_df))
    (RUN_DIR / "warnings.json").write_text(json.dumps(warnings, indent=2))
    (RUN_DIR / "fee_source.json").write_text(json.dumps({
        "url": "https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf",
        "revision_date": None,
        "extracted_at": None,
        "modeled_categories": [],
        "excluded_categories": ["SEC", "FINRA_TAF", "FINRA_CAT", "ORF", "OCC", "commissions"],
    }, indent=2))

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
