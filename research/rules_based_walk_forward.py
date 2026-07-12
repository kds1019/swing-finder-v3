"""
Backtests the RULES-BASED screening/trade-plan system on its own — no ML anywhere in this
script. Every walk-forward test run so far this session (research/walk_forward_backtest.py,
research/triple_barrier_walk_forward.py) asked whether an ML layer improves on the base
system; none of them asked whether the base system — core.smartscore.compute_smartscore's
technical setup classification plus core.trade_plan.compute_trade_plan's stop/target —
has a real edge on its own. This answers that.

Two separate questions, both answered here:
1. Does the trade plan itself have positive expected value — does price hit target before
   stop often enough, relative to each trade's own R:R ratio, to be profitable on average?
2. Does SmartScore's own ranking actually separate good setups from bad ones — do
   higher-SmartScore trade plans win more / return more than lower-SmartScore ones?

Reuses core.trade_plan.resolve_trade_plan_outcome() — the same function
core/pick_tracking.py uses to resolve live picks and research/triple_barrier_walk_forward.py
used for its classifier's labels — so "did this trade work" means the same thing
everywhere in this codebase. Same MAX_HOLD_DAYS=30 vertical barrier as live tracking.

Scope note (same simplification already accepted in research/walk_forward_backtest.py):
this tests compute_smartscore + compute_trade_plan per ticker in isolation, not the full
live pipeline's cross-sectional sector-cap/deep-discount adjustments, which require
scanning the whole universe simultaneously on each historical date — out of scope for a
per-ticker walk-forward pass.

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. No FMP/News dependency at all — this system is
100% price/volume-derived, so nothing here needs external data beyond bars.

Usage:
    python -m research.rules_based_walk_forward
    python -m research.rules_based_walk_forward --n-tickers 60 --lookback-days 760 \
        --step-days 3 --max-hold-days 30 --output research/rules_based_results.csv

--target-mode flat isolation test (added after the 2026-07-12 real-data run found
significant negative expectancy at 10.55:1 average R:R and 5.4% win rate — a signature of
targets set too far for MAX_HOLD_DAYS, not necessarily bad entries): overrides
compute_trade_plan()'s Fibonacci-extension target with a flat settings.min_risk_reward
multiple of the *same* stop/risk compute_trade_plan already computed. Entry and stop are
completely unchanged — only the target-distance variable is isolated, to answer whether
SmartScore's entry-timing has real value once the target isn't set unrealistically far.
Not a change to core.trade_plan.compute_trade_plan() itself (production behavior is
untouched) — this experimental override lives only in this research script.

    python -m research.rules_based_walk_forward --target-mode flat \
        --output research/rules_based_results_flat_target.csv

Output CSV columns: as_of_date, ticker, smartscore, setup, near_miss, entry, stop, target,
rr_ratio, weak_rr, direction_correct (True = target hit before stop, named to match
research/analyze_confidence.py::confidence_bucket_report()'s expected column so that
function can be reused unmodified), actual_return_pct, bars_to_resolution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agents.market_data_agent import MarketDataAgent
from config.settings import load_settings
from core.indicators import compute_indicators
from core.pick_tracking import MAX_HOLD_DAYS
from core.smartscore import compute_smartscore
from core.trade_plan import compute_trade_plan, resolve_trade_plan_outcome
from core.universe import load_universe
from research.walk_forward_backtest import WARMUP_BARS, select_sample_universe

RESULT_COLUMNS = [
    "as_of_date", "ticker", "smartscore", "setup", "near_miss", "entry", "stop", "target",
    "rr_ratio", "weak_rr", "direction_correct", "actual_return_pct", "bars_to_resolution",
]

TARGET_MODES = ("fibonacci", "flat")


def apply_flat_target(plan: dict, settings) -> dict | None:
    """Experimental override for the --target-mode flat isolation test: same entry/stop as
    compute_trade_plan() (its swing-low/EMA-anchored stop logic isn't implicated in the
    negative-expectancy finding — only the Fibonacci-extension target is), but the target
    becomes a flat settings.min_risk_reward multiple of that same risk distance, not an
    extension that can imply many multiples more. Returns None if risk is zero/invalid
    (mirrors compute_trade_plan's own None-on-insufficient-data contract)."""
    entry, stop = plan["entry"], plan["stop"]
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    target = round(entry + settings.min_risk_reward * risk, 2)
    return {
        "entry": entry, "stop": stop, "target": target,
        "rr_ratio": settings.min_risk_reward, "weak_rr": False,
        "stop_distance_sanity_flag": plan["stop_distance_sanity_flag"], "fib_warning": "",
    }


def backtest_ticker_rules(
    ticker: str, df: pd.DataFrame, settings, max_hold_days: int, step_days: int,
    target_mode: str = "fibonacci",
) -> list[dict]:
    """Walk df forward step_days at a time; at each point, score and plan the trade using
    only data up to that bar (df.iloc[:idx+1] — indicators computed causally over the full
    history up front, so slicing is equivalent to recomputing fresh, same reasoning as
    research/walk_forward_backtest.py::backtest_ticker), then resolve the outcome against
    the actual (already historical) bars that follow. Skips any as-of date where
    compute_smartscore finds no setup or near-miss — the live pipeline never computes a
    trade plan for those either (core.market_data_agent.MarketDataAgent.scan_universe only
    calls compute_trade_plan after a ticker already has a non-None smartscore)."""
    rows = []
    last_idx = len(df) - 1
    for idx in range(WARMUP_BARS, last_idx + 1, step_days):
        df_upto = df.iloc[: idx + 1].reset_index(drop=True)
        score_result = compute_smartscore(df_upto, settings)
        if score_result.get("smartscore") is None:
            continue

        plan = compute_trade_plan(df_upto, settings)
        if plan is None:
            continue
        if target_mode == "flat":
            plan = apply_flat_target(plan, settings)
            if plan is None:
                continue

        after = df.iloc[idx + 1: idx + 1 + max_hold_days].reset_index(drop=True)
        if len(after) < max_hold_days:
            continue  # not enough forward history yet to resolve a definitive label

        outcome, outcome_price, _outcome_date, bars = resolve_trade_plan_outcome(
            after, plan["stop"], plan["target"], max_hold_days
        )
        if outcome not in ("target_hit", "stop_hit"):
            continue  # drop expired_unresolved — ambiguous, not a clean win/loss

        entry = plan["entry"]
        rows.append({
            "as_of_date": str(pd.to_datetime(df["Date"].iloc[idx]).date()),
            "ticker": ticker,
            "smartscore": score_result["smartscore"],
            "setup": score_result["setup"],
            "near_miss": score_result["near_miss"],
            "entry": entry,
            "stop": plan["stop"],
            "target": plan["target"],
            "rr_ratio": plan["rr_ratio"],
            "weak_rr": plan["weak_rr"],
            "direction_correct": outcome == "target_hit",
            "actual_return_pct": round((outcome_price - entry) / entry * 100, 2),
            "bars_to_resolution": bars,
        })
    return rows


def run(
    n_tickers: int, lookback_days: int, step_days: int, max_hold_days: int, seed: int,
    output: str, target_mode: str = "fibonacci",
) -> None:
    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[rules_based] sampled {len(tickers)} tickers (same seed as walk_forward_backtest.py "
          f"for apples-to-apples comparison), target_mode={target_mode}", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[rules_based] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < WARMUP_BARS + max_hold_days + 20:
            print(f"[rules_based] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        df = compute_indicators(df.copy())
        rows = backtest_ticker_rules(ticker, df, settings, max_hold_days, step_days, target_mode)
        print(f"[rules_based] ({i}/{len(tickers)}) {ticker}: {len(rows)} scored trade plans", file=sys.stderr)
        total_rows += len(rows)

        if rows:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True

    print(f"[rules_based] done — {total_rows} total scored trade plans written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=3)
    parser.add_argument("--max-hold-days", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="research/rules_based_results.csv")
    parser.add_argument("--target-mode", type=str, choices=TARGET_MODES, default="fibonacci")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.step_days, args.max_hold_days, args.seed,
        args.output, args.target_mode)


if __name__ == "__main__":
    main()
