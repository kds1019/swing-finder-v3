"""
Tests whether core.patterns.detect_patterns()'s 8 chart-pattern detectors (Bull
Flag, Cup and Handle, Double Bottom, Ascending Triangle, Bear Flag, Double Top, Head
and Shoulders, Descending Triangle) actually track forward returns — the last of
SmartScore's four score-adjustment inputs (setup classification, ML edge, volume
profile, chart patterns) that hadn't been walk-forward tested this research effort.
Setup classification and ML edge already tested null; volume profile is untested and
separately tracked; this closes out the pattern-detection piece.

Same deconfounding principle as research/smartscore_walk_forward.py: measures a raw
N-day-ahead close-to-close return, no target/stop/barrier resolution at all, plus a
no-signal control population (dates where detect_patterns() found nothing) as a
baseline. Unlike SmartScore's audit, patterns have an explicit directional bias
("Bullish" or "Bearish") — evaluate_pattern_score() rewards Bullish patterns and
penalizes Bearish ones, and core.patterns.py's own "action" text says to buy Bullish
breakouts and avoid longs on Bearish ones. So `direction_correct` here is bias-aware:
a Bullish pattern is "correct" if the forward return is positive, a Bearish pattern is
"correct" if the forward return is negative — a stricter, more specific test than
SmartScore's "any signal beats no signal," since these detectors make an explicit
directional claim that can be checked directly.

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. No FMP/News dependency — pure price data.

Usage:
    python -m research.patterns_walk_forward
    python -m research.patterns_walk_forward --n-tickers 60 --lookback-days 760 \
        --step-days 10 --days-ahead 30

Output CSV columns: as_of_date, ticker, has_signal, pattern_name, pattern_bias,
pattern_confidence, entry_price, actual_price, actual_return_pct, direction_correct
(bias-aware as described above — named to match
research/analyze_confidence.py::confidence_bucket_report()'s expected column so that
function can be reused unmodified with column="pattern_confidence").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agents.market_data_agent import MarketDataAgent
from config.settings import load_settings
from core.indicators import compute_indicators
from core.patterns import detect_patterns
from core.pick_tracking import MAX_HOLD_DAYS
from core.universe import load_universe
from research.walk_forward_backtest import WARMUP_BARS, select_sample_universe

RESULT_COLUMNS = [
    "as_of_date", "ticker", "has_signal", "pattern_name", "pattern_bias",
    "pattern_confidence", "entry_price", "actual_price", "actual_return_pct",
    "direction_correct",
]


def backtest_ticker_patterns(ticker: str, df: pd.DataFrame, days_ahead: int, step_days: int) -> list[dict]:
    """Walk df forward step_days at a time; at each point, detect patterns using only
    data up to that bar (df.iloc[:idx+1] — same causal-slicing reasoning as every other
    walk-forward script this session), then record the actual days_ahead-bar-later raw
    return, which is already known since this is historical data. No target/stop
    resolution — this measures the pattern signal alone. detect_patterns() has no
    price/volume gating of its own (unlike compute_smartscore), so every step produces
    a row — either a detected pattern (has_signal=True) or the no-signal control
    (has_signal=False)."""
    rows = []
    last_idx = len(df) - days_ahead - 1
    for idx in range(WARMUP_BARS, last_idx + 1, step_days):
        df_upto = df.iloc[: idx + 1].reset_index(drop=True)
        patterns = detect_patterns(df_upto)

        entry_price = float(df["Close"].iloc[idx])
        actual_price = float(df["Close"].iloc[idx + days_ahead])
        actual_return_pct = round((actual_price - entry_price) / entry_price * 100, 2)
        as_of_date = str(pd.to_datetime(df["Date"].iloc[idx]).date())

        if patterns:
            top = patterns[0]
            bias = top["bias"]
            direction_correct = actual_return_pct > 0 if bias == "Bullish" else actual_return_pct < 0
            rows.append({
                "as_of_date": as_of_date, "ticker": ticker, "has_signal": True,
                "pattern_name": top["type"], "pattern_bias": bias,
                "pattern_confidence": top["confidence"], "entry_price": entry_price,
                "actual_price": actual_price, "actual_return_pct": actual_return_pct,
                "direction_correct": direction_correct,
            })
        else:
            rows.append({
                "as_of_date": as_of_date, "ticker": ticker, "has_signal": False,
                "pattern_name": None, "pattern_bias": None, "pattern_confidence": None,
                "entry_price": entry_price, "actual_price": actual_price,
                "actual_return_pct": actual_return_pct, "direction_correct": actual_return_pct > 0,
            })
    return rows


def run(n_tickers: int, lookback_days: int, step_days: int, days_ahead: int, seed: int, output: str) -> None:
    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[patterns] sampled {len(tickers)} tickers (same seed as walk_forward_backtest.py "
          f"for apples-to-apples comparison)", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[patterns] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < WARMUP_BARS + days_ahead + 20:
            print(f"[patterns] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        df = compute_indicators(df.copy())
        rows = backtest_ticker_patterns(ticker, df, days_ahead, step_days)
        print(f"[patterns] ({i}/{len(tickers)}) {ticker}: {len(rows)} scored dates", file=sys.stderr)
        total_rows += len(rows)

        if rows:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True

    print(f"[patterns] done — {total_rows} total scored dates written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--days-ahead", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="research/patterns_results.csv")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.step_days, args.days_ahead, args.seed, args.output)


if __name__ == "__main__":
    main()
