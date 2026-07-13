"""
Analysis for research/patterns_walk_forward.py's output — do core.patterns.py's 8
chart-pattern detectors actually track forward returns?

Four checks:
1. Signal vs. no-signal baseline: do detected-pattern dates beat a no-signal control
   population (bias-aware — Bullish patterns should beat control on the upside,
   Bearish patterns should beat it on the downside)?
2. Rank-IC (Spearman) between pattern_confidence and forward return, computed
   separately for Bullish and Bearish patterns, since they predict opposite
   directions — a working Bullish detector should show positive rank-IC (higher
   confidence, more upside); a working Bearish detector should show negative rank-IC
   (higher confidence, more downside). Reuses
   research/analyze_smartscore.py::compute_rank_ic() unmodified.
3. Calibration: reuses research/analyze_confidence.py::confidence_bucket_report()
   unmodified (column="pattern_confidence") — direction_correct is already bias-aware
   from research/patterns_walk_forward.py, so a rising win rate by confidence bucket
   means the score is doing its job regardless of which patterns land in which bucket.
4. By pattern type (all 8 detectors + no-signal control): does each individual
   detector's bias-aware win rate and mean return actually support its own directional
   claim?

Usage:
    python -m research.analyze_patterns
    python -m research.analyze_patterns --input research/patterns_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from research.analyze_confidence import confidence_bucket_report
from research.analyze_smartscore import compute_rank_ic


def run(input_path: str) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze_patterns] {input_path} is empty — nothing to analyze.")
        return
    print(f"[analyze_patterns] {len(df)} scored dates loaded from {input_path}\n")

    signal_df = df[df["has_signal"]].dropna(subset=["pattern_confidence", "actual_return_pct"])
    control_df = df[~df["has_signal"]].dropna(subset=["actual_return_pct"])
    bullish_df = signal_df[signal_df["pattern_bias"] == "Bullish"]
    bearish_df = signal_df[signal_df["pattern_bias"] == "Bearish"]

    print("=== Signal vs. no-signal baseline (bias-aware win rate) ===")
    for label, group in (
        ("all patterns (Bullish + Bearish)", signal_df),
        ("Bullish patterns only", bullish_df),
        ("Bearish patterns only", bearish_df),
        ("no-signal control", control_df),
    ):
        if group.empty:
            print(f"  {label}: n=0")
            continue
        n = len(group)
        win_rate = round(float(group["direction_correct"].mean()) * 100, 1)
        mean_ret = round(float(group["actual_return_pct"].mean()), 3)
        print(f"  {label}: n={n} win_rate_pct={win_rate} mean_return_pct={mean_ret}")
    print(
        "\n  Interpretation: win_rate_pct for Bullish/Bearish groups is already bias-aware "
        "(a Bearish pattern 'wins' when the forward return is negative) — compare against "
        "the no-signal control's plain positive-return rate. If patterns aren't beating "
        "control by a real margin, core.patterns.py's detectors aren't adding value over "
        "doing nothing, regardless of how they rank internally below."
    )

    print("\n=== Rank-IC: pattern_confidence vs. forward return, by bias ===")
    for label, group in (("Bullish", bullish_df), ("Bearish", bearish_df)):
        ic = compute_rank_ic(group["pattern_confidence"], group["actual_return_pct"])
        print(f"  {label}: {ic}")
    print(
        "\n  Interpretation: a working Bullish detector should show positive rank-IC "
        "(higher confidence tracks more upside); a working Bearish detector should show "
        "negative rank-IC (higher confidence tracks more downside). Near-zero or "
        "non-significant in either direction means confidence isn't tracking outcome "
        "magnitude, even if the plain win rate above looks reasonable."
    )

    print("\n=== Confidence bucket report (quintiles, bias-aware win rate) ===")
    buckets = confidence_bucket_report(signal_df, n_buckets=5, column="pattern_confidence")
    print(buckets.to_string(index=False))

    print("\n=== By pattern type (all 8 detectors + no-signal control) ===")
    labeled = df.copy()
    labeled["pattern_label"] = np.where(~labeled["has_signal"], "no_signal_control", labeled["pattern_name"])
    pattern_report = labeled.dropna(subset=["actual_return_pct"]).groupby("pattern_label").apply(
        lambda g: pd.Series({
            "n": len(g),
            "win_rate_pct": round(float(g["direction_correct"].mean()) * 100, 1),
            "mean_actual_return_pct": round(float(g["actual_return_pct"].mean()), 3),
        }),
        include_groups=False,
    )
    print(pattern_report.to_string())
    print(
        "\n  Interpretation: win_rate_pct is bias-aware per pattern (Bullish patterns "
        "measured against positive returns, Bearish against negative). Each detector "
        "should clear the no_signal_control row by a real margin, and should clear ~50% "
        "on its own terms — a Bearish pattern with a low bias-aware win rate means price "
        "went up more often than down after that pattern fired, the opposite of its own "
        "claimed action."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/patterns_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
