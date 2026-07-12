"""
The triple-barrier reframe from docs/ml-edge-confidence-research.md's "Plan (not yet
built)" section: after the stale-features bug fix erased the regression ensemble's only
validated edge (see that doc's 2026-07-11 update), this is the "next genuinely different
idea" — stop predicting a continuous 5-day return and instead predict something tied
directly to what actually determines a trade's outcome: given this specific entry/stop/
target (core.trade_plan.compute_trade_plan), does price hit target before stop?

This is Lopez de Prado's triple-barrier labeling method applied as the *primary* model's
target, not meta-labeling on top of an existing regression — it replaces what the model
predicts, it doesn't add a second model on top of core.ml_forecast's ensemble.

Label generation reuses core.trade_plan.resolve_trade_plan_outcome() — the exact same
function core/pick_tracking.py uses to resolve live picks — so a validated result
generalizes to the live pipeline's own definition of "did this trade work," not a bespoke
backtest-only one. Feature engineering reuses core.ml_forecast.build_feature_table(), the
same technical/fundamental feature set the regression ensemble already trains on (RS/
weekly-trend/VWAP, insider, rating, VIX), just paired with this different label. Model is
LGBMClassifier instead of RandomForestRegressor/GradientBoostingRegressor — same per-ticker
training shape as core.ml_forecast's forecasters (one time-based train/test split per
ticker, not a walk-forward retrain-at-every-step loop like research/walk_forward_backtest.py
uses for the regression ensemble; each row here already encodes its own historical as-of
date via compute_trade_plan on a truncated df, so a single split is sufficient and far
cheaper than retraining per step).

Requires ALPACA_API_KEY/ALPACA_SECRET_KEY. FMP_API_KEY is optional (insider/rating
features skipped if unset, same as research/walk_forward_backtest.py). News-sentiment
history is fetched via Alpaca (no separate credential) and FinBERT-scored once per ticker
before any per-ticker training, same reasoning as research/walk_forward_backtest.py.

Usage:
    python -m research.triple_barrier_walk_forward
    python -m research.triple_barrier_walk_forward --n-tickers 60 --lookback-days 760 \
        --step-days 3 --max-hold-days 30 --output research/triple_barrier_results.csv

Output CSV columns: as_of_date, ticker, entry, stop, target, rr_ratio, p_target,
direction_correct (True = target hit before stop), actual_return_pct, bars_to_resolution.

Also writes a second CSV (--importances-output, default
research/triple_barrier_feature_importances.csv) of each feature's LGBM importance,
averaged across every ticker's own trained classifier — feature-level attribution the
walk-forward regression script doesn't capture (see docs/ml-edge-confidence-research.md's
2026-07-12 update on why that mattered for analyst_revision_net_90d).
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
from core.ml_forecast import build_feature_table
from core.pick_tracking import MAX_HOLD_DAYS
from core.sentiment import build_sentiment_df
from core.trade_plan import compute_trade_plan, resolve_trade_plan_outcome
from core.universe import load_universe
from research.walk_forward_backtest import select_sample_universe

LABEL_COLUMNS = [
    "as_of_date", "ticker", "entry", "stop", "target", "rr_ratio",
    "bars_to_resolution", "direction_correct", "actual_return_pct",
]
RESULT_COLUMNS = [
    "as_of_date", "ticker", "entry", "stop", "target", "rr_ratio", "p_target",
    "direction_correct", "actual_return_pct", "bars_to_resolution",
]

# Below this many labeled (target_hit/stop_hit) examples, a per-ticker train/test split is
# too small to trust either the trained classifier or its held-out evaluation.
MIN_LABELED_ROWS = 100


def build_ticker_dataset(
    ticker: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None,
    insider_df: pd.DataFrame | None,
    rating_df: pd.DataFrame | None,
    grades_df: pd.DataFrame | None,
    sentiment_df: pd.DataFrame | None,
    settings,
    max_hold_days: int,
    step_days: int,
) -> tuple[pd.DataFrame, list[str]]:
    """One row per historical as-of date with enough trailing history (feature rolling-
    window warmup) and enough trailing... rather, *forward*, history (max_hold_days bars)
    to resolve a definitive triple-barrier label. Sampled every step_days bars — computing
    a trade plan at every single bar is not needed for a one-shot train/test split (unlike
    the regression ensemble's per-step retrain) and find_support_resistance's per-call cost
    scales with how much history it's given, so a dense per-bar sweep across ~700 bars would
    be wastefully slow.

    Each row's entry/stop/target come from compute_trade_plan(df truncated to that date) —
    the identical function and stop/target logic the live pipeline uses, not a synthetic
    label — and its outcome from resolve_trade_plan_outcome() walking forward through the
    (already historical, already known) bars that follow. expired_unresolved rows (neither
    level touched within max_hold_days) are dropped: ambiguous, not a clean binary label.
    """
    features, dates_aligned = build_feature_table(
        df, vix_df=None, spy_df=spy_df, insider_df=insider_df, rating_df=rating_df,
        grades_df=grades_df, sentiment_df=sentiment_df, lookback=1500,
    )
    if features is None:
        return pd.DataFrame(), []

    features = features.drop(columns=["close"])
    feature_names = features.columns.tolist()

    date_col = pd.to_datetime(df["Date"]).dt.normalize()
    date_to_idx = {d: i for i, d in enumerate(date_col)}

    rows = []
    for pos in range(0, len(features), step_days):
        as_of_date = dates_aligned[pos]
        df_idx = date_to_idx.get(as_of_date)
        if df_idx is None:
            continue

        after = df.iloc[df_idx + 1: df_idx + 1 + max_hold_days].reset_index(drop=True)
        if len(after) < max_hold_days:
            continue  # not enough forward history yet to resolve a definitive label

        df_upto = df.iloc[: df_idx + 1].reset_index(drop=True)
        plan = compute_trade_plan(df_upto, settings)
        if plan is None:
            continue

        outcome, outcome_price, _outcome_date, bars = resolve_trade_plan_outcome(
            after, plan["stop"], plan["target"], max_hold_days
        )
        if outcome not in ("target_hit", "stop_hit"):
            continue

        entry = plan["entry"]
        row = {
            "as_of_date": as_of_date,
            "ticker": ticker,
            "entry": entry,
            "stop": plan["stop"],
            "target": plan["target"],
            "rr_ratio": plan["rr_ratio"],
            "bars_to_resolution": bars,
            "direction_correct": outcome == "target_hit",
            "actual_return_pct": round((outcome_price - entry) / entry * 100, 2),
        }
        row.update(zip(feature_names, features.iloc[pos].values))
        rows.append(row)

    return pd.DataFrame(rows), feature_names


def run(
    n_tickers: int, lookback_days: int, step_days: int, max_hold_days: int,
    seed: int, train_frac: float, output: str, importances_output: str,
) -> None:
    from lightgbm import LGBMClassifier

    settings = load_settings()
    universe_df = load_universe(settings.universe_csv_path)
    tickers = select_sample_universe(universe_df, settings, n_tickers, seed)
    print(f"[triple_barrier] sampled {len(tickers)} tickers (same seed as "
          f"walk_forward_backtest.py for apples-to-apples comparison)", file=sys.stderr)

    agent = MarketDataAgent(settings)
    bars_by_ticker = agent.fetch_universe_bars(tickers, lookback_days=lookback_days)
    print(f"[triple_barrier] fetched bars for {len(bars_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    spy_df = agent.fetch_spy_bars(lookback_days=lookback_days)
    print(f"[triple_barrier] SPY bars: "
          f"{'fetched ' + str(len(spy_df)) + ' rows' if spy_df is not None else 'unavailable — RS features will be skipped'}",
          file=sys.stderr)

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
                print(f"[triple_barrier] {ticker}: FMP insider/rating/grades fetch failed ({e}), "
                      f"skipping those features for this ticker", file=sys.stderr)
        print(f"[triple_barrier] fetched insider/rating/grades data for "
              f"{sum(1 for t in tickers if t in insider_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)
    else:
        print("[triple_barrier] FMP_API_KEY not set — insider/rating/grades features will be skipped", file=sys.stderr)

    # News sentiment: fetched and FinBERT-scored once per ticker here, same reasoning as
    # research/walk_forward_backtest.py — no settings gate needed since ALPACA_API_KEY is
    # already required for this whole script.
    sentiment_by_ticker: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            news_df = agent.fetch_news(ticker, lookback_days=lookback_days)
            sentiment_by_ticker[ticker] = build_sentiment_df(news_df)
        except Exception as e:
            print(f"[triple_barrier] {ticker}: news fetch/sentiment scoring failed ({e}), "
                  f"skipping that feature for this ticker", file=sys.stderr)
    print(f"[triple_barrier] scored news sentiment for "
          f"{sum(1 for t in tickers if t in sentiment_by_ticker)}/{len(tickers)} tickers", file=sys.stderr)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = output_path.exists()

    total_test_rows = 0
    importance_rows = []
    for i, ticker in enumerate(tickers, 1):
        df = bars_by_ticker.get(ticker)
        if df is None or len(df) < 150:
            print(f"[triple_barrier] ({i}/{len(tickers)}) {ticker}: insufficient history, skipping", file=sys.stderr)
            continue

        df = compute_indicators(df.copy())
        dataset, feature_names = build_ticker_dataset(
            ticker, df, spy_df, insider_by_ticker.get(ticker), rating_by_ticker.get(ticker),
            grades_by_ticker.get(ticker), sentiment_by_ticker.get(ticker), settings, max_hold_days, step_days,
        )
        if len(dataset) < MIN_LABELED_ROWS:
            print(f"[triple_barrier] ({i}/{len(tickers)}) {ticker}: only {len(dataset)} labeled "
                  f"examples, skipping", file=sys.stderr)
            continue

        dataset = dataset.sort_values("as_of_date").reset_index(drop=True)
        split_idx = int(len(dataset) * train_frac)
        train, test = dataset.iloc[:split_idx], dataset.iloc[split_idx:]
        if len(test) < 10 or train["direction_correct"].nunique() < 2:
            print(f"[triple_barrier] ({i}/{len(tickers)}) {ticker}: not enough test rows or "
                  f"single-class train set, skipping", file=sys.stderr)
            continue

        clf = LGBMClassifier(
            n_estimators=200, max_depth=4, min_child_samples=20, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        clf.fit(train[feature_names].values, train["direction_correct"].values)
        p_target = clf.predict_proba(test[feature_names].values)[:, 1]

        result = test[LABEL_COLUMNS].copy()
        result["p_target"] = np.round(p_target, 4)
        total_test_rows += len(result)

        importance_rows.extend(
            {"ticker": ticker, "feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, clf.feature_importances_)
        )

        print(f"[triple_barrier] ({i}/{len(tickers)}) {ticker}: {len(train)} train / {len(test)} test "
              f"({dataset['direction_correct'].mean() * 100:.1f}% target-hit-first overall)", file=sys.stderr)

        result[RESULT_COLUMNS].to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

    print(f"[triple_barrier] done — {total_test_rows} total held-out predictions written to {output_path}",
          file=sys.stderr)

    if importance_rows:
        # LGBM's feature_importances_ (default "split" type: how many times a feature was
        # used to split, not gain) isn't directly comparable across tickers with different
        # tree counts/depths reached — a ticker whose trees happened to split more overall
        # would dominate a raw average regardless of whether any single feature actually
        # mattered more for it. Normalize each ticker's importances to sum to 1 first (a
        # per-ticker relative-importance distribution), then average those normalized
        # shares across tickers — the same spirit as research/pooled_model_experiment.py's
        # single-model importances, just aggregated over N per-ticker models instead of one
        # pooled one, without letting tree-complexity differences skew the ranking.
        imp_df = pd.DataFrame(importance_rows)
        imp_df["norm_importance"] = (
            imp_df.groupby("ticker")["importance"].transform(lambda s: s / s.sum() if s.sum() else s)
        )
        summary = (
            imp_df.groupby("feature")["norm_importance"]
            .agg(mean_importance="mean", n_tickers="count")
            .sort_values("mean_importance", ascending=False)
            .reset_index()
        )
        imp_path = Path(importances_output)
        imp_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(imp_path, index=False)
        print(f"\n[triple_barrier] mean per-ticker-normalized LGBM feature importances across "
              f"{imp_df['ticker'].nunique()} trained tickers, written to {imp_path}:", file=sys.stderr)
        for _, row in summary.head(15).iterrows():
            print(f"  {row['feature']}: {row['mean_importance']:.4f} (n={int(row['n_tickers'])})", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-tickers", type=int, default=60)
    parser.add_argument("--lookback-days", type=int, default=760)
    parser.add_argument("--step-days", type=int, default=3)
    parser.add_argument("--max-hold-days", type=int, default=MAX_HOLD_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--output", type=str, default="research/triple_barrier_results.csv")
    parser.add_argument("--importances-output", type=str,
                         default="research/triple_barrier_feature_importances.csv")
    args = parser.parse_args()
    run(args.n_tickers, args.lookback_days, args.step_days, args.max_hold_days,
        args.seed, args.train_frac, args.output, args.importances_output)


if __name__ == "__main__":
    main()
