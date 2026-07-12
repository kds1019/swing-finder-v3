"""
Tests whether core.smartscore.compute_smartscore's setup classification and its
individual scoring components actually correlate with forward returns —
deconfounded from target/stop mechanics entirely. research/rules_based_walk_forward.py
already showed target distance was a real, separate problem (three target formulas
tried, none profitable); this script asks a different question that no test this
session has asked directly: is the *entry* signal itself — classify_setup()'s
Breakout/Pullback thresholds and compute_smartscore's setup_strength/trend/volume/
base/level/fibonacci bonus weights — any good, independent of how a target is set?

Measures a raw N-day-ahead close-to-close return (no stop, no target, no barrier
resolution at all) so a null result here can't be blamed on target distance, and a
positive result would mean the setup classification has real, independently testable
predictive value regardless of exit logic.

Also records a "no signal" control population — same tickers/dates, but where
compute_smartscore found no setup and no near-miss (reason="no_signal") — so
Breakout/Pullback results have an actual baseline to beat, not just each other.
("filtered"/"insufficient_data" reasons are excluded from the control group: those are
data-quality exclusions — fails price/volume sanity or too little history — not "no
trading interest," so they aren't a meaningful comparison population.)

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. No FMP/News dependency — like
rules_based_walk_forward.py, this is 100% price/volume-derived.

Usage:
    python -m research.smartscore_walk_forward
    python -m research.smartscore_walk_forward --n-tickers 60 --lookback-days 760 \
        --step-days 10 --days-ahead 30 --output research/smartscore_results.csv

Output CSV columns: as_of_date, ticker, smartscore, setup, near_miss, has_signal,
setup_strength, trend_context, volume_bonus, base_tightness_bonus,
meaningful_level_bonus, fibonacci_bonus, entry_price, actual_price, actual_return_pct,
direction_correct (True if actual_return_pct > 0 — named to match
research/analyze_confidence.py::confidence_bucket_report()'s expected column so that
function can be reused unmodified with column="smartscore").
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
from core.universe import load_universe
from research.walk_forward_backtest import WARMUP_BARS, select_sample_universe

RESULT_COLUMNS = [
    "as_of_date", "ticker", "smartscore", "setup", "near_miss", "has_signal",
    "setup_strength", "trend_context", "volume_bonus", "base_tightness_bonus",
    "meaningful_level_bonus", "fibonacci_bonus", "entry_price", "actual_price",
    "actual_return_pct", "direction_correct",
]

# reasons compute_smartscore returns that mean "not a valid comparison point at all"
# (data-quality exclusions), as opposed to "no_signal" (a real control-group member —
# passed filters, just no Breakout/Pullback/near-miss pattern found).
EXCLUDED_REASONS = ("insufficient_data", "filtered")


def backtest_ticker_smartscore(
    ticker: str, df: pd.DataFrame, settings, days_ahead: int, step_days: int,
) -> list[dict]:
    """Walk df forward step_days at a time; at each point, score using only data up to
    that bar (df.iloc[:idx+1] — indicators computed causally over the full history up
    front, so slicing is equivalent to recomputing fresh, same reasoning as every other
    walk-forward script this session), then record the actual days_ahead-bar-later raw
    return, which is already known since this is historical data. No target/stop
    resolution at all — this measures the entry signal alone."""
    rows = []
    last_idx = len(df) - days_ahead - 1
    for idx in range(WARMUP_BARS, last_idx + 1, step_days):
        df_upto = df.iloc[: idx + 1].reset_index(drop=True)
        result = compute_smartscore(df_upto, settings)
        if result.get("reason") in EXCLUDED_REASONS:
            continue

        entry_price = float(df["Close"].iloc[idx])
        actual_price = float(df["Close"].iloc[idx + days_ahead])
        actual_return_pct = round((actual_price - entry_price) / entry_price * 100, 2)
        breakdown = result.get("breakdown") or {}

        rows.append({
            "as_of_date": str(pd.to_datetime(df["Date"].iloc[idx]).date()),
            "ticker": ticker,
            "smartscore": result.get("smartscore"),
            "setup": result.get("setup"),
            "near_miss": result.get("near_miss", False),
            "has_signal": result.get("smartscore") is not None,
            "setup_strength": breakdown.get("setup_strength"),
            "trend_context": breakdown.get("trend_context"),
            "volume_bonus": breakdown.get("volume"),
            "base_tightness_bonus": breakdown.get("base_tightness"),
            "meaningful_level_bonus": breakdown.get("meaningful_level"),
            "fibonacci_bonus": breakdown.get("fibonacci"),
            "entry_price": entry_price,
            "actual_price": actual_price,
            "actual_return_pct": actual_return_pct,
            "direction_correct": actual_return_pct > 0,
        })
    return rows


def run(n_tickers: int, lookback_days: int, step_days: int, days_ahead: int, seed: int, output: str) -> None:
    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[smartscore] sampled {len(tickers)} tickers (same seed as walk_forward_backtest.py "
          f"for apples-to-apples comparison)", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[smartscore] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < WARMUP_BARS + days_ahead + 20:
            print(f"[smartscore] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        df = compute_indicators(df.copy())
        rows = backtest_ticker_smartscore(ticker, df, settings, days_ahead, step_days)
        print(f"[smartscore] ({i}/{len(tickers)}) {ticker}: {len(rows)} scored dates", file=sys.stderr)
        total_rows += len(rows)

        if rows:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True

    print(f"[smartscore] done — {total_rows} total scored dates written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--days-ahead", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="research/smartscore_results.csv")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.step_days, args.days_ahead, args.seed, args.output)


if __name__ == "__main__":
    main()
