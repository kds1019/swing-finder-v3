"""
ML forecast accuracy tracker.

Persists every ML ensemble forecast (core.ml_forecast.ensemble_ml_forecast) made on the
shortlist to a durable, append-only log (committed to the repo alongside results/, since
GitHub Actions runners have no persistent local disk between runs) so forecast quality can
be measured against what actually happened, rather than trusted purely from a single run's
train/test R^2. Each pipeline run: score any past predictions whose 5-trading-day window has
now elapsed (fetch the actual close via Alpaca), then log this run's new forecasts for future
scoring.

"5 trading days ahead" is resolved by counting bars in the fetched window, not calendar-day
arithmetic — matching how ensemble_ml_forecast itself defines days_ahead (a bar-count shift,
not a calendar shift), so scoring stays consistent with how the model was trained.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LOG_COLUMNS = [
    "prediction_date", "ticker", "entry_price", "predicted_return_pct", "predicted_price",
    "confidence", "days_ahead", "scored", "actual_price", "actual_return_pct",
    "direction_correct", "scored_date",
]

# Safety net only — every run attempts to score every outstanding prediction, so this should
# rarely trigger. Guards against a ticker going permanently dark (delisted, data gap) leaving
# a row unscored forever.
MAX_UNSCORED_AGE_DAYS = 45

MIN_SAMPLE_SIZE = 15
ACCURACY_WINDOW = 60  # most recent N scored predictions considered "recent track record"


def load_predictions_log(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(p)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[LOG_COLUMNS]


def save_predictions_log(log_df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(path, index=False)


def record_predictions(log_df: pd.DataFrame, new_predictions: list[dict]) -> pd.DataFrame:
    """new_predictions: list of {prediction_date, ticker, entry_price, predicted_return_pct,
    predicted_price, confidence, days_ahead} — one per successful ML forecast this run."""
    if not new_predictions:
        return log_df

    rows = [{
        "prediction_date": p["prediction_date"],
        "ticker": p["ticker"],
        "entry_price": p["entry_price"],
        "predicted_return_pct": p["predicted_return_pct"],
        "predicted_price": p["predicted_price"],
        "confidence": p["confidence"],
        "days_ahead": p.get("days_ahead", 5),
        "scored": False,
        "actual_price": None,
        "actual_return_pct": None,
        "direction_correct": None,
        "scored_date": None,
    } for p in new_predictions]

    return pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True)


def score_due_predictions(log_df: pd.DataFrame, market_agent) -> pd.DataFrame:
    """
    Fills in actual_price/actual_return_pct/direction_correct for any unscored row whose
    days_ahead-bar window has now elapsed. market_agent is a MarketDataAgent, used to fetch
    a short recent bars window per outstanding ticker (batched into one call).
    """
    if log_df.empty:
        return log_df

    unscored_mask = log_df["scored"] != True  # noqa: E712 - explicit bool compare, NaN-safe
    unscored = log_df[unscored_mask]
    if unscored.empty:
        return log_df

    log_df = log_df.copy()
    today = pd.Timestamp.now().normalize()

    tickers = unscored["ticker"].dropna().unique().tolist()
    bars_by_ticker = market_agent.fetch_universe_bars(tickers, lookback_days=40)

    for idx, row in unscored.iterrows():
        ticker = row["ticker"]
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty:
            continue

        bars = bars.reset_index(drop=True)
        bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
        pred_date = pd.to_datetime(row["prediction_date"]).normalize()
        days_ahead = int(row["days_ahead"]) if pd.notna(row["days_ahead"]) else 5

        on_or_after = bars[bars["Date"] >= pred_date]
        if on_or_after.empty:
            # Prediction date isn't covered by this 40-trading-day window — either the
            # window is stale relative to a very old prediction, or data is missing.
            age_days = (today - pred_date).days
            if age_days > MAX_UNSCORED_AGE_DAYS:
                log_df.loc[idx, ["scored", "scored_date"]] = [True, str(today.date())]
            continue

        pred_bar_idx = on_or_after.index[0]
        target_bar_idx = pred_bar_idx + days_ahead
        if target_bar_idx >= len(bars):
            continue  # not enough trading days have elapsed yet — try again next run

        actual_price = float(bars.loc[target_bar_idx, "Close"])
        entry_price = float(row["entry_price"])
        actual_return_pct = round((actual_price - entry_price) / entry_price * 100, 2)

        predicted_return_pct = row["predicted_return_pct"]
        direction_correct = None
        if pd.notna(predicted_return_pct):
            direction_correct = (float(predicted_return_pct) > 0) == (actual_return_pct > 0)

        log_df.loc[idx, "scored"] = True
        log_df.loc[idx, "actual_price"] = actual_price
        log_df.loc[idx, "actual_return_pct"] = actual_return_pct
        log_df.loc[idx, "direction_correct"] = direction_correct
        log_df.loc[idx, "scored_date"] = str(today.date())

    return log_df


def compute_accuracy_summary(log_df: pd.DataFrame, min_sample: int = MIN_SAMPLE_SIZE) -> dict:
    """
    Rolling directional-accuracy summary over the most recent ACCURACY_WINDOW scored
    predictions. sufficient_data=False below min_sample tells callers (the Decision Agent
    prompt) not to draw conclusions from too little history yet.
    """
    scored = log_df[(log_df["scored"] == True) & log_df["direction_correct"].notna()]  # noqa: E712
    scored = scored.tail(ACCURACY_WINDOW)
    sample_size = len(scored)

    if sample_size < min_sample:
        return {"sufficient_data": False, "sample_size": sample_size, "min_sample_size": min_sample}

    correct = scored["direction_correct"].astype(bool)
    directional_accuracy_pct = round(correct.mean() * 100, 1)

    errors = (scored["actual_return_pct"].astype(float) - scored["predicted_return_pct"].astype(float)).abs()
    mean_abs_error_pct = round(errors.mean(), 2)

    return {
        "sufficient_data": True,
        "sample_size": sample_size,
        "directional_accuracy_pct": directional_accuracy_pct,
        "mean_abs_error_pct": mean_abs_error_pct,
    }
