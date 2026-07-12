"""
Offline walk-forward backtest for the ML ensemble's confidence output.

Phase 1 of docs/ml-edge-confidence-research.md: the live pipeline's
ml_predictions.csv log only grows by one row per shortlisted ticker per manual
run, so it'll take months to accumulate enough scored predictions to say
anything statistically meaningful about whether `confidence` tracks real
forward-return accuracy. This script generates that data offline instead, by
re-running the exact same modeling code (core.ml_forecast.ensemble_ml_forecast)
at many historical "as of" dates across a sample of the universe, using data
that would have been available as of each date — no lookahead.

Scope note: this measures the ML ensemble's general calibration (does its
confidence track forward-return accuracy across a representative sample of the
universe over time), not calibration conditioned on "this ticker was on the
SmartScore shortlist that day" — reproducing the full historical SmartScore/
sector-cap/deep-discount pipeline at every walk-forward date across two years
was out of scope for this pass. That's a reasonable proxy since the ensemble's
confidence formula itself never references SmartScore.

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY (same as the live pipeline) — this is
a real historical-data pull, not something that can run without credentials.
FMP_API_KEY is optional — if set, also fetches insider-trading and daily
quant-rating history per ticker (core.ml_forecast's insider_df/rating_df
features); if unset, those features are skipped and everything else runs the
same as before. News-sentiment history (core.ml_forecast's sentiment_df,
FinBERT-scored via core.sentiment) is fetched per ticker via Alpaca's News API —
no separate credential needed since ALPACA_API_KEY is already required — and
scored once per ticker before any walk-forward looping, since FinBERT inference
is comparatively expensive and every walk-forward step for one ticker shares the
same underlying news history.

Usage:
    python -m research.walk_forward_backtest
    python -m research.walk_forward_backtest --n-tickers 60 --lookback-days 760 \
        --step-days 10 --days-ahead 5 --output research/walk_forward_results.csv

Output CSV columns: as_of_date, ticker, entry_price, predicted_price,
predicted_return_pct, confidence, rf_confidence, gb_confidence, rf_r2, gb_r2,
agreement_pct, actual_price, actual_return_pct, direction_correct.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agents.market_data_agent import MarketDataAgent
from agents.research_agent import ResearchAgent
from config.settings import load_settings
from core.indicators import compute_indicators
from core.ml_forecast import ensemble_ml_forecast
from core.sentiment import build_sentiment_df
from core.universe import load_universe

RESULT_COLUMNS = [
    "as_of_date", "ticker", "entry_price", "predicted_price", "predicted_return_pct",
    "confidence", "rf_confidence", "gb_confidence", "rf_r2", "gb_r2", "agreement_pct",
    "actual_price", "actual_return_pct", "direction_correct",
]

# Bars needed before the first walk-forward evaluation point — comfortably past where
# EMA200/rolling-20 windows and prepare_features' own internal rolling features (up to a
# 20-bar momentum lookback) have all stabilized.
WARMUP_BARS = 300


def select_sample_universe(universe_df: pd.DataFrame, settings, n: int, seed: int) -> list[str]:
    """Sector-proportional sample, restricted to the price/volume band the live pipeline
    actually screens (settings.price_min/max, settings.min_volume) — a random sample of
    the full universe would include tickers (penny stocks, mega-caps) the live shortlist
    would never surface, which would calibrate confidence against a population the
    pipeline doesn't actually trade."""
    candidates = universe_df[
        universe_df["Price"].between(settings.price_min, settings.price_max)
        & (universe_df["Volume"] >= settings.min_volume)
    ]
    if candidates.empty:
        candidates = universe_df

    sectors = sorted(candidates["Sector"].dropna().unique())
    rng = np.random.RandomState(seed)

    per_sector = max(1, n // max(1, len(sectors)))
    picked: list[str] = []
    for sector in sectors:
        pool = candidates.loc[candidates["Sector"] == sector, "Ticker"].tolist()
        rng.shuffle(pool)
        picked.extend(pool[:per_sector])

    if len(picked) < n:
        remaining = [t for t in candidates["Ticker"].tolist() if t not in picked]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])

    return sorted(picked[:n])


def backtest_ticker(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None,
    insider_df: pd.DataFrame | None,
    rating_df: pd.DataFrame | None,
    grades_df: pd.DataFrame | None,
    sentiment_df: pd.DataFrame | None,
    step_days: int,
    days_ahead: int,
) -> list[dict]:
    """Walk df forward step_days at a time; at each point, train the ensemble on data up
    to that bar only (df.iloc[:idx+1] — indicators were computed causally over the full
    history up front, so slicing is equivalent to recomputing them fresh at each step) and
    score it against the actual close days_ahead bars later, which is already known since
    this is historical data. spy_df/insider_df/rating_df/grades_df/sentiment_df are
    truncated to the same as-of date at each step for the same no-lookahead reason
    (prepare_features' own reindexing already bounds this, but truncating here too is
    cheap defense-in-depth). sentiment_df is already FinBERT-scored by the caller before
    this function is ever called — only the cheap rolling-window aggregation happens
    per-step here, not per-step rescoring."""
    rows = []
    last_idx = len(df) - days_ahead - 1
    for idx in range(WARMUP_BARS, last_idx + 1, step_days):
        df_upto = df.iloc[: idx + 1].reset_index(drop=True)
        as_of_date = df["Date"].iloc[idx]
        spy_upto = spy_df[spy_df["Date"] <= as_of_date] if spy_df is not None else None
        insider_upto = insider_df[insider_df["filingDate"] <= as_of_date] if insider_df is not None else None
        rating_upto = rating_df[rating_df["Date"] <= as_of_date] if rating_df is not None else None
        grades_upto = grades_df[grades_df["date"] <= as_of_date] if grades_df is not None else None
        sentiment_upto = sentiment_df[sentiment_df["Date"] <= as_of_date] if sentiment_df is not None else None
        result = ensemble_ml_forecast(
            df_upto, vix_df=None, spy_df=spy_upto, insider_df=insider_upto, rating_df=rating_upto,
            grades_df=grades_upto, sentiment_df=sentiment_upto, days_ahead=days_ahead,
        )
        if not result.get("success"):
            continue

        entry_price = float(df["Close"].iloc[idx])
        actual_price = float(df["Close"].iloc[idx + days_ahead])
        actual_return_pct = round((actual_price - entry_price) / entry_price * 100, 2)
        predicted_return_pct = round((result["ensemble_price"] - entry_price) / entry_price * 100, 2)

        rows.append({
            "as_of_date": str(pd.to_datetime(df["Date"].iloc[idx]).date()),
            "ticker": ticker,
            "entry_price": entry_price,
            "predicted_price": result["ensemble_price"],
            "predicted_return_pct": predicted_return_pct,
            "confidence": result["confidence"],
            "rf_confidence": result["rf_confidence"],
            "gb_confidence": result["gb_confidence"],
            "rf_r2": result["rf_r2"],
            "gb_r2": result["gb_r2"],
            "agreement_pct": result["agreement"],
            "actual_price": actual_price,
            "actual_return_pct": actual_return_pct,
            "direction_correct": (predicted_return_pct > 0) == (actual_return_pct > 0),
        })
    return rows


def run(n_tickers: int, lookback_days: int, step_days: int, days_ahead: int, seed: int, output: str) -> None:
    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[walk_forward] sampled {len(tickers)} tickers across "
          f"{universe_df['Sector'].nunique()} sectors: {tickers}", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[walk_forward] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    spy_df = agent.fetch_spy_bars(lookback_days=lookback_days)
    print(f"[walk_forward] SPY bars: {'fetched ' + str(len(spy_df)) + ' rows' if spy_df is not None else 'unavailable — RS features will be skipped'}", file=sys.stderr)

    insider_by_ticker: dict[str, pd.DataFrame] = {}
    rating_by_ticker: dict[str, pd.DataFrame] = {}
    grades_by_ticker: dict[str, pd.DataFrame] = {}
    if settings.fmp_api_key:
        research_agent = ResearchAgent(settings)
        for ticker in tickers:
            try:
                insider_by_ticker[ticker] = research_agent.get_insider_trades(ticker)
                rating_by_ticker[ticker] = research_agent.get_rating_history(ticker)
                grades_by_ticker[ticker] = research_agent.get_grade_history(ticker)
            except Exception as e:
                print(f"[walk_forward] {ticker}: FMP insider/rating/grades fetch failed ({e}), "
                      f"skipping those features for this ticker", file=sys.stderr)
        print(f"[walk_forward] fetched insider/rating/grades data for "
              f"{sum(1 for t in tickers if t in insider_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)
    else:
        print("[walk_forward] FMP_API_KEY not set — insider/rating/grades features will be skipped", file=sys.stderr)

    # News sentiment: fetched and FinBERT-scored once per ticker here, up front — not
    # inside backtest_ticker's per-step loop, since scoring is comparatively expensive and
    # every walk-forward step for a given ticker shares the same underlying news history.
    # No settings gate (unlike insider/rating/grades above) since ALPACA_API_KEY is already
    # required for this whole script.
    sentiment_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            news_df = agent.fetch_news(ticker, lookback_days=lookback_days)
            sentiment_by_ticker[ticker] = build_sentiment_df(news_df)
        except Exception as e:
            print(f"[walk_forward] {ticker}: news fetch/sentiment scoring failed ({e}), "
                  f"skipping that feature for this ticker", file=sys.stderr)
    print(f"[walk_forward] scored news sentiment for "
          f"{sum(1 for t in tickers if t in sentiment_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < WARMUP_BARS + days_ahead + 20:
            print(f"[walk_forward] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        df = compute_indicators(df.copy())
        rows = backtest_ticker(
            ticker, df, spy_df, insider_by_ticker.get(ticker), rating_by_ticker.get(ticker),
            grades_by_ticker.get(ticker), sentiment_by_ticker.get(ticker), step_days, days_ahead
        )
        print(f"[walk_forward] ({i}/{len(tickers)}) {ticker}: {len(rows)} walk-forward predictions", file=sys.stderr)
        total_rows += len(rows)

        if rows:
            pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )
            wrote_header = True

    print(f"[walk_forward] done — {total_rows} total predictions written to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--days-ahead", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="research/walk_forward_results.csv")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.step_days, args.days_ahead, args.seed, args.output)


if __name__ == "__main__":
    main()
