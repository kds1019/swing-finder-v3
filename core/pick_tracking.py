"""
Pick outcome tracker.

Persists every Decision Agent ranked pick (ticker/rank/smartscore/entry/stop/target/rr_ratio
— the actual final recommendation, not just the raw ML forecast core.ml_tracking covers) to
a durable, append-only log (pick_outcomes.csv, committed to the repo alongside results/ and
ml_predictions.csv). Each run, before generating new picks: walk forward through Alpaca bars
since each unresolved pick's date, checking High/Low against its stop/target to determine
which was hit first. This answers "when this system says rank 1 / SmartScore 90, does it
actually work out" — a question about the pipeline's own decision quality, independent of
whether any given pick was actually traded (that's the user's own journal's job).

Same-bar ambiguity (a daily bar's range touches both stop and target) can't be sequenced from
OHLC data alone — resolved conservatively toward stop_hit, since assuming the better outcome
would overstate win rate.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LOG_COLUMNS = [
    "prediction_date", "ticker", "rank", "smartscore", "entry_price", "stop_price",
    "target_price", "rr_ratio", "resolved", "outcome", "outcome_price", "outcome_date",
    "bars_to_resolution", "actual_return_pct",
]

# Swing trades are meant to resolve in days-to-weeks, not months. A pick that hasn't hit
# either stop or target within this many trading days is marked "expired_unresolved" rather
# than tracked open forever — inconclusive, not a failure.
MAX_HOLD_DAYS = 30

MIN_SAMPLE_SIZE = 10
ACCURACY_WINDOW = 60  # most recent N resolved (decisive) picks considered "recent track record"


def load_pick_outcomes_log(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(p)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[LOG_COLUMNS]


def save_pick_outcomes_log(log_df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(path, index=False)


def record_picks(log_df: pd.DataFrame, ranked_picks: list[dict], prediction_date: str) -> pd.DataFrame:
    """ranked_picks: DecisionAgent.synthesize()'s own result["ranked_picks"] list — already
    has ticker/rank/smartscore/entry/stop/target/rr_ratio per pick, used as-is."""
    if not ranked_picks:
        return log_df

    rows = [{
        "prediction_date": prediction_date,
        "ticker": p["ticker"],
        "rank": p["rank"],
        "smartscore": p["smartscore"],
        "entry_price": p["entry"],
        "stop_price": p["stop"],
        "target_price": p["target"],
        "rr_ratio": p["rr_ratio"],
        "resolved": False,
        "outcome": None,
        "outcome_price": None,
        "outcome_date": None,
        "bars_to_resolution": None,
        "actual_return_pct": None,
    } for p in ranked_picks]

    return pd.concat([log_df, pd.DataFrame(rows)], ignore_index=True)


def score_due_picks(log_df: pd.DataFrame, market_agent) -> pd.DataFrame:
    """
    Resolves any unresolved pick whose stop or target has since been touched (checked via
    High/Low on each bar since the pick's date, in chronological order — first level touched
    wins), or that has aged past MAX_HOLD_DAYS without either being touched.
    """
    if log_df.empty:
        return log_df

    unresolved_mask = log_df["resolved"] != True  # noqa: E712 - explicit bool compare, NaN-safe
    unresolved = log_df[unresolved_mask]
    if unresolved.empty:
        return log_df

    log_df = log_df.copy()
    tickers = unresolved["ticker"].dropna().unique().tolist()
    bars_by_ticker = market_agent.fetch_universe_bars(tickers, lookback_days=MAX_HOLD_DAYS + 10)

    for idx, row in unresolved.iterrows():
        ticker = row["ticker"]
        bars = bars_by_ticker.get(ticker)
        if bars is None or bars.empty:
            continue

        bars = bars.reset_index(drop=True)
        bars["Date"] = pd.to_datetime(bars["Date"]).dt.normalize()
        pred_date = pd.to_datetime(row["prediction_date"]).normalize()

        after = bars[bars["Date"] > pred_date].reset_index(drop=True)
        if after.empty:
            continue  # no new bars since the pick yet

        stop = float(row["stop_price"])
        target = float(row["target_price"])
        entry_price = float(row["entry_price"])

        outcome = None
        outcome_price = None
        outcome_date = None
        bars_checked = 0

        for i in range(min(len(after), MAX_HOLD_DAYS)):
            bar = after.iloc[i]
            bars_checked = i + 1
            hit_stop = bar["Low"] <= stop
            hit_target = bar["High"] >= target
            if hit_stop:
                # Checked before target on purpose — see module docstring on same-bar tie-break.
                outcome, outcome_price, outcome_date = "stop_hit", stop, bar["Date"]
                break
            if hit_target:
                outcome, outcome_price, outcome_date = "target_hit", target, bar["Date"]
                break

        if outcome is not None:
            actual_return_pct = round((outcome_price - entry_price) / entry_price * 100, 2)
            log_df.loc[idx, ["resolved", "outcome", "outcome_price", "outcome_date",
                              "bars_to_resolution", "actual_return_pct"]] = [
                True, outcome, outcome_price, str(pd.Timestamp(outcome_date).date()),
                bars_checked, actual_return_pct,
            ]
        elif len(after) >= MAX_HOLD_DAYS:
            last_bar = after.iloc[MAX_HOLD_DAYS - 1]
            last_close = float(last_bar["Close"])
            actual_return_pct = round((last_close - entry_price) / entry_price * 100, 2)
            log_df.loc[idx, ["resolved", "outcome", "outcome_price", "outcome_date",
                              "bars_to_resolution", "actual_return_pct"]] = [
                True, "expired_unresolved", last_close, str(last_bar["Date"].date()),
                MAX_HOLD_DAYS, actual_return_pct,
            ]
        # else: still open, not enough bars have elapsed yet — leave unresolved for next run

    return log_df


def compute_pick_accuracy_summary(log_df: pd.DataFrame, min_sample: int = MIN_SAMPLE_SIZE) -> dict:
    """
    Rolling win-rate summary over the most recent ACCURACY_WINDOW *decisively* resolved picks
    (target_hit or stop_hit — expired_unresolved picks are excluded from win rate since they
    never actually resolved either way, though they still count toward general awareness).
    sufficient_data=False below min_sample tells callers (the Decision Agent prompt) not to
    draw conclusions from too little history yet.
    """
    resolved = log_df[log_df["resolved"] == True]  # noqa: E712
    decisive = resolved[resolved["outcome"].isin(["target_hit", "stop_hit"])].tail(ACCURACY_WINDOW)
    sample_size = len(decisive)

    if sample_size < min_sample:
        return {"sufficient_data": False, "sample_size": sample_size, "min_sample_size": min_sample}

    win_rate_pct = round((decisive["outcome"] == "target_hit").mean() * 100, 1)
    avg_bars_to_resolution = round(decisive["bars_to_resolution"].astype(float).mean(), 1)
    avg_return_pct = round(decisive["actual_return_pct"].astype(float).mean(), 2)

    rank1 = decisive[decisive["rank"] == 1]
    rank1_win_rate_pct = round((rank1["outcome"] == "target_hit").mean() * 100, 1) if len(rank1) >= 5 else None

    return {
        "sufficient_data": True,
        "sample_size": sample_size,
        "win_rate_pct": win_rate_pct,
        "avg_bars_to_resolution": avg_bars_to_resolution,
        "avg_return_pct": avg_return_pct,
        "rank1_win_rate_pct": rank1_win_rate_pct,
    }
