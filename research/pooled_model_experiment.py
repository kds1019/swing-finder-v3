"""
Punch-list item #1 from docs/ml-edge-confidence-research.md: does pooling
training data across tickers — one shared model instead of one per ticker —
improve on the per-ticker walk-forward backtest's rank-IC (0.0436, the best
result so far, from research/walk_forward_backtest.py after adding the RS/
weekly-trend/VWAP features)?

Reuses the same ticker sample (select_sample_universe, seed=42) as
walk_forward_backtest.py for an apples-to-apples comparison. Instead of
retraining a fresh model at every historical as-of date (that script's
approach — 1 model per ticker per date), this builds ONE pooled dataset from
every ticker's full available history via prepare_features(), trains ONE
shared RF + GBM on it with a time-based (not random, not per-ticker) train/
test split, and evaluates IC/rank-IC on the held-out period.

Most of prepare_features' engineered columns are already scale-normalized
(ratios to the ticker's own close, percentage returns, booleans) rather than
raw price levels — that's what makes pooling rows from tickers at wildly
different price points into one training set valid in the first place.

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. FMP_API_KEY is optional (insider/rating/grades
features skipped if unset, same as research/walk_forward_backtest.py).

Also serves a second purpose beyond the pooling-vs-per-ticker comparison above: a single
pooled model gives one ranked feature-importance list over a much larger combined sample
than any individual ticker's own few-hundred-row model would, which is otherwise not
captured anywhere — research/walk_forward_backtest.py's per-step retraining never logs
importances, and this script already computed but only ever printed the top 10. See
docs/ml-edge-confidence-research.md's 2026-07-12 update for why that gap mattered for
evaluating whether a newly added feature (e.g. analyst_revision_net_90d) is being used at
all, not just whether it moved the ensemble's overall accuracy.

Usage:
    python -m research.pooled_model_experiment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from agents.market_data_agent import MarketDataAgent
from agents.research_agent import ResearchAgent
from config.settings import load_settings
from core.indicators import compute_indicators
from core.ml_forecast import prepare_features
from core.universe import load_universe
from research.walk_forward_backtest import select_sample_universe

MIN_TICKER_BARS = 120


def build_pooled_dataset(
    tickers: list[str],
    bars_by_ticker: dict,
    spy_df: pd.DataFrame | None,
    insider_by_ticker: dict,
    rating_by_ticker: dict,
    grades_by_ticker: dict,
    days_ahead: int,
) -> pd.DataFrame:
    """One row per (ticker, as-of date) sample — the panel dataset a cross-sectional
    model trains on, instead of one ticker's own few-hundred-row time series."""
    frames = []
    canonical_features: list[str] | None = None

    for ticker in tickers:
        bars = bars_by_ticker.get(ticker)
        if bars is None or len(bars) < MIN_TICKER_BARS:
            continue

        df = compute_indicators(bars.copy())
        X, y, feature_names, _, dates, _current_features = prepare_features(
            df, vix_df=None, spy_df=spy_df,
            insider_df=insider_by_ticker.get(ticker), rating_df=rating_by_ticker.get(ticker),
            grades_df=grades_by_ticker.get(ticker), days_ahead=days_ahead,
        )
        if X is None:
            continue

        if canonical_features is None:
            canonical_features = feature_names
        elif feature_names != canonical_features:
            # Different optional columns present (e.g. weekly_uptrend/rs_* gated behind
            # minimum-history checks) — skip rather than silently concatenating mismatched
            # columns under the same column index.
            print(f"[pooled] {ticker}: feature set differs from the rest of the pool "
                  f"({len(feature_names)} vs {len(canonical_features)} columns), skipping", file=sys.stderr)
            continue

        frame = pd.DataFrame(X, columns=feature_names)
        frame["forward_return"] = y
        frame["as_of_date"] = pd.to_datetime(dates)
        frame["ticker"] = ticker
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def time_based_split(pooled: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by calendar date, not row count or per-ticker position — every training row
    must be strictly before every test row, regardless of which ticker it came from, or a
    later-dated row from ticker A could leak into an earlier-dated test window via a
    different ticker's row ordering."""
    cutoff = pooled["as_of_date"].quantile(train_frac)
    train = pooled[pooled["as_of_date"] < cutoff]
    test = pooled[pooled["as_of_date"] >= cutoff]
    return train, test


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    pearson_ic, pearson_p = stats.pearsonr(y_pred, y_true)
    spearman_ic, spearman_p = stats.spearmanr(y_pred, y_true)
    direction_correct = (y_pred > 0) == (y_true > 0)
    print(f"\n=== {label} ===")
    print(f"  n={len(y_true)}")
    print(f"  ic_pearson={pearson_ic:.4f} p={pearson_p:.4f}")
    print(f"  rank_ic_spearman={spearman_ic:.4f} p={spearman_p:.4f}")
    print(f"  directional_accuracy_pct={round(float(direction_correct.mean()) * 100, 1)}")


def run(
    n_tickers: int, lookback_days: int, days_ahead: int, seed: int, train_frac: float,
    importances_output: str,
) -> None:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[pooled] sampled {len(tickers)} tickers (same seed as walk_forward_backtest.py "
          f"for apples-to-apples comparison)", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    spy_df = agent.fetch_spy_bars(lookback_days=lookback_days)
    print(f"[pooled] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

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
                print(f"[pooled] {ticker}: FMP insider/rating/grades fetch failed ({e}), "
                      f"skipping those features for this ticker", file=sys.stderr)
        print(f"[pooled] fetched insider/rating/grades data for "
              f"{sum(1 for t in tickers if t in insider_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)
    else:
        print("[pooled] FMP_API_KEY not set — insider/rating/grades features will be skipped", file=sys.stderr)

    pooled = build_pooled_dataset(
        tickers, bars_by_ticker, spy_df, insider_by_ticker, rating_by_ticker, grades_by_ticker, days_ahead
    )
    if pooled.empty:
        print("[pooled] no usable data — aborting", file=sys.stderr)
        return
    print(f"[pooled] pooled dataset: {len(pooled)} rows from "
          f"{pooled['ticker'].nunique()} tickers, {pooled.shape[1] - 3} features", file=sys.stderr)

    train, test = time_based_split(pooled, train_frac)
    print(f"[pooled] time-based split: {len(train)} train rows (< {train['as_of_date'].max().date()}), "
          f"{len(test)} test rows (>= {test['as_of_date'].min().date()})", file=sys.stderr)

    feature_cols = [c for c in pooled.columns if c not in ("forward_return", "as_of_date", "ticker")]
    X_train, y_train = train[feature_cols].values, train["forward_return"].values
    X_test, y_test = test[feature_cols].values, test["forward_return"].values

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=20, max_features=0.5, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    evaluate(y_test, rf_pred, "Pooled Random Forest — held-out test period")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    gb = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, min_samples_leaf=20, learning_rate=0.05, subsample=0.8, random_state=42,
    )
    gb.fit(X_train_scaled, y_train)
    gb_pred = gb.predict(X_test_scaled)
    evaluate(y_test, gb_pred, "Pooled Gradient Boosting — held-out test period")

    ensemble_pred = (rf_pred + gb_pred) / 2
    evaluate(y_test, ensemble_pred, "Pooled ensemble (RF+GB average) — held-out test period")

    importances = pd.DataFrame({
        "feature": feature_cols, "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importances.insert(0, "rank", importances.index + 1)

    imp_path = Path(importances_output)
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    importances.to_csv(imp_path, index=False)

    print(f"\n=== Top 10 RF feature importances (pooled model, full ranking written to "
          f"{imp_path}) ===")
    for _, row in importances.head(10).iterrows():
        print(f"  {int(row['rank'])}. {row['feature']}: {row['importance']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--days-ahead", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--importances-output", type=str,
                         default="research/pooled_feature_importances.csv")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.days_ahead, args.seed, args.train_frac,
        args.importances_output)


if __name__ == "__main__":
    main()
