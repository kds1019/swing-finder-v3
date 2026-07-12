"""
Tests cross-sectional momentum — one of the most replicated factors in equity
research — as a candidate replacement for core.smartscore's setup classification,
which the SmartScore audit (docs/ml-edge-confidence-research.md, "SmartScore audit"
update) found no evidence for: the signal population underperformed a no-signal
control on every measure. This tests a completely independent ranking signal with
real academic backing, deconfounded from SmartScore/classify_setup entirely — no
setup detection, no target/stop mechanics, just a ticker's own trailing return
against a raw forward return.

Uses the standard "12-1 momentum" formation (Jegadeesh & Titman): trailing return
over `--momentum-lookback-days` (default 252 trading days, ~12 months), skipping the
most recent `--momentum-skip-days` (default 21, ~1 month) to avoid the well-documented
short-term reversal effect that shows up if the skip is omitted.

Scope note (simplification also used by every rank-IC test this session —
research/analyze_confidence.py::compute_ic, research/analyze_smartscore.py's
compute_rank_ic): this is a pooled panel rank-IC across all ticker/date pairs, not a
strict date-by-date cross-sectional ranking normalized within each trading day.
Academic momentum studies usually rank stocks against each other within the same
date; pooling across dates is a reasonable, simpler proxy consistent with how every
other factor test this session was done, but a stock market-wide regime shift during
the sample window could bias a pooled test in a way a true cross-sectional design
would cancel out. Worth strengthening if this signal shows promise.

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. No FMP/News dependency — pure price data.

Usage:
    python -m research.momentum_walk_forward
    python -m research.momentum_walk_forward --n-tickers 60 --lookback-days 760 \
        --step-days 10 --days-ahead 30 --momentum-lookback-days 252 --momentum-skip-days 21

Output CSV columns: as_of_date, ticker, momentum_return_pct, entry_price, actual_price,
actual_return_pct, direction_correct (True if actual_return_pct > 0 — named to match
research/analyze_confidence.py::confidence_bucket_report()'s expected column so that
function can be reused unmodified with column="momentum_return_pct").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agents.market_data_agent import MarketDataAgent
from config.settings import load_settings
from core.pick_tracking import MAX_HOLD_DAYS
from core.universe import load_universe
from research.walk_forward_backtest import WARMUP_BARS, select_sample_universe

RESULT_COLUMNS = [
    "as_of_date", "ticker", "momentum_return_pct", "entry_price", "actual_price",
    "actual_return_pct", "direction_correct",
]


def backtest_ticker_momentum(
    ticker: str, df: pd.DataFrame, days_ahead: int, step_days: int,
    momentum_lookback_days: int, momentum_skip_days: int,
) -> list[dict]:
    """Walk df forward step_days at a time; at each point, compute trailing momentum
    using only data up to that bar (no lookahead — formation window ends
    momentum_skip_days before the as-of date, entirely in the past), then record the
    actual days_ahead-bar-later raw return, which is already known since this is
    historical data. No setup classification, no target/stop resolution — this
    measures momentum alone."""
    rows = []
    formation_bars = momentum_lookback_days + momentum_skip_days
    min_idx = max(WARMUP_BARS, formation_bars)
    last_idx = len(df) - days_ahead - 1
    for idx in range(min_idx, last_idx + 1, step_days):
        formation_end_idx = idx - momentum_skip_days
        formation_start_idx = formation_end_idx - momentum_lookback_days
        if formation_start_idx < 0:
            continue

        price_start = float(df["Close"].iloc[formation_start_idx])
        price_end = float(df["Close"].iloc[formation_end_idx])
        if price_start <= 0:
            continue
        momentum_return_pct = round((price_end - price_start) / price_start * 100, 2)

        entry_price = float(df["Close"].iloc[idx])
        actual_price = float(df["Close"].iloc[idx + days_ahead])
        actual_return_pct = round((actual_price - entry_price) / entry_price * 100, 2)

        rows.append({
            "as_of_date": str(pd.to_datetime(df["Date"].iloc[idx]).date()),
            "ticker": ticker,
            "momentum_return_pct": momentum_return_pct,
            "entry_price": entry_price,
            "actual_price": actual_price,
            "actual_return_pct": actual_return_pct,
            "direction_correct": actual_return_pct > 0,
        })
    return rows


def run(
    n_tickers: int, lookback_days: int, step_days: int, days_ahead: int,
    momentum_lookback_days: int, momentum_skip_days: int, seed: int, output: str,
) -> None:
    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[momentum] sampled {len(tickers)} tickers (same seed as walk_forward_backtest.py "
          f"for apples-to-apples comparison)", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[momentum] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    min_history = max(WARMUP_BARS, momentum_lookback_days + momentum_skip_days) + days_ahead + 20
    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < min_history:
            print(f"[momentum] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        rows = backtest_ticker_momentum(
            ticker, df, days_ahead, step_days, momentum_lookback_days, momentum_skip_days
        )
        print(f"[momentum] ({i}/{len(tickers)}) {ticker}: {len(rows)} scored dates", file=sys.stderr)
        total_rows += len(rows)

        if rows:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True

    print(f"[momentum] done — {total_rows} total scored dates written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--days-ahead", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--momentum-lookback-days", type=int, default=252)
    parser.add_argument("--momentum-skip-days", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="research/momentum_results.csv")
    args = parser.parse_args()
    run(
        args.n_tickers, args.lookback_days, args.step_days, args.days_ahead,
        args.momentum_lookback_days, args.momentum_skip_days, args.seed, args.output,
    )


if __name__ == "__main__":
    main()
