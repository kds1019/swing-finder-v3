"""
Analysis for research/smartscore_walk_forward.py's output — do
core.smartscore.compute_smartscore's setup classification and scoring components
actually correlate with forward returns, deconfounded from target/stop mechanics?

Three checks, all new relative to every prior analysis script this session (which all
evaluated either an ML layer or the trade plan's target/stop, never the entry signal
alone against a raw forward return):

1. Signal vs. no-signal baseline: do Breakout/Pullback/near-miss dates actually
   outperform a same-tickers/same-dates control population where compute_smartscore
   found nothing? This is the population-level check no other test this session ran —
   without it, a within-signal comparison (e.g. Breakout vs Pullback) can't tell you
   whether *either* beats doing nothing.
2. Rank-IC (Spearman) between smartscore and forward return, plus a rank-IC breakdown
   per individual scoring component (setup_strength, trend_context, volume, base
   tightness, meaningful level, fibonacci zone) — does each ingredient in the score
   actually track forward returns on its own, or is compute_smartscore summing several
   components with no real predictive value into one that only looks meaningful?
3. Calibration: reuses research/analyze_confidence.py::confidence_bucket_report()
   unmodified (column="smartscore") — the output CSV's direction_correct/
   actual_return_pct columns were named to match that function's existing contract for
   exactly this reason.

Usage:
    python -m research.analyze_smartscore
    python -m research.analyze_smartscore --input research/smartscore_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from research.analyze_confidence import confidence_bucket_report

MIN_ROWS_FOR_CORRELATION = 3  # scipy's spearmanr/pearsonr are undefined below this

# breakdown component -> output CSV column, in the order compute_smartscore applies them
COMPONENT_COLUMNS = {
    "setup_strength": "setup_strength",
    "trend_context": "trend_context",
    "volume": "volume_bonus",
    "base_tightness": "base_tightness_bonus",
    "meaningful_level": "meaningful_level_bonus",
    "fibonacci": "fibonacci_bonus",
}


def compute_rank_ic(series: pd.Series, actual_return_pct: pd.Series) -> dict:
    valid = pd.DataFrame({"x": series, "y": actual_return_pct}).dropna()
    if len(valid) < MIN_ROWS_FOR_CORRELATION or valid["x"].nunique() < 2:
        return {"n": len(valid), "rank_ic": None, "p_value": None, "note": "too few distinct values"}
    rank_ic, p_value = stats.spearmanr(valid["x"], valid["y"])
    return {"n": len(valid), "rank_ic": round(float(rank_ic), 4), "p_value": round(float(p_value), 4)}


def run(input_path: str) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze_smartscore] {input_path} is empty — nothing to analyze.")
        return
    print(f"[analyze_smartscore] {len(df)} scored dates loaded from {input_path}\n")

    signal_df = df[df["has_signal"]].dropna(subset=["smartscore", "actual_return_pct"])
    control_df = df[~df["has_signal"]].dropna(subset=["actual_return_pct"])

    print("=== Signal vs. no-signal baseline ===")
    for label, group in (("signal (Breakout/Pullback/near-miss)", signal_df), ("no-signal control", control_df)):
        if group.empty:
            print(f"  {label}: n=0")
            continue
        n = len(group)
        win_rate = round(float(group["direction_correct"].mean()) * 100, 1)
        mean_ret = round(float(group["actual_return_pct"].mean()), 3)
        median_ret = round(float(group["actual_return_pct"].median()), 3)
        print(f"  {label}: n={n} win_rate_pct={win_rate} mean_return_pct={mean_ret} median_return_pct={median_ret}")
    print(
        "\n  Interpretation: if the signal population's win rate / mean return isn't "
        "meaningfully better than the no-signal control's, compute_smartscore's setup "
        "classification isn't adding value over doing nothing, regardless of how the "
        "signal population ranks internally below."
    )

    print("\n=== Rank-IC: smartscore vs. forward return (signal population only) ===")
    ic = compute_rank_ic(signal_df["smartscore"], signal_df["actual_return_pct"])
    for k, v in ic.items():
        print(f"  {k}: {v}")
    print(
        "\n  Interpretation: near-zero or non-significant rank-IC means a higher "
        "SmartScore doesn't track a better forward return at all, independent of "
        "whatever the signal-vs-control comparison above showed."
    )

    print("\n=== Rank-IC per scoring component (signal population only) ===")
    for component, column in COMPONENT_COLUMNS.items():
        comp_ic = compute_rank_ic(signal_df[column], signal_df["actual_return_pct"])
        print(f"  {component} ({column}): {comp_ic}")
    print(
        "\n  Interpretation: compute_smartscore sums several bonus components into one "
        "score. A component with a near-zero/non-significant rank-IC here is adding "
        "noise (or worse) to the total rather than real signal, even if the combined "
        "smartscore's own rank-IC above looks non-trivial."
    )

    print("\n=== SmartScore bucket report (quintiles, signal population only) ===")
    buckets = confidence_bucket_report(signal_df, n_buckets=5, column="smartscore")
    print(buckets.to_string(index=False))

    print("\n=== By setup type (Breakout / Pullback / near-miss-only / no-signal control) ===")
    labeled = df.copy()
    labeled["setup_label"] = np.where(
        ~labeled["has_signal"], "no_signal_control", labeled["setup"].fillna("near_miss_only")
    )
    setup_report = labeled.dropna(subset=["actual_return_pct"]).groupby("setup_label").apply(
        lambda g: pd.Series({
            "n": len(g),
            "win_rate_pct": round(float(g["direction_correct"].mean()) * 100, 1),
            "mean_actual_return_pct": round(float(g["actual_return_pct"].mean()), 3),
        }),
        include_groups=False,
    )
    print(setup_report.to_string())
    print(
        "\n  Interpretation: if classify_setup()'s thresholds are doing their job, "
        "Breakout and Pullback should both beat no_signal_control, ideally by a "
        "meaningful margin — not just outrank each other."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/smartscore_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
