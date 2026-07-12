"""
Analysis for research/momentum_walk_forward.py's output — does cross-sectional
momentum (trailing 12-1 return) correlate with forward returns, as a candidate
replacement for core.smartscore's setup classification (which the SmartScore audit
found no evidence for)?

Two checks:
1. Rank-IC (Spearman) between momentum_return_pct and forward return — reuses
   research/analyze_smartscore.py::compute_rank_ic() unmodified (it's already
   column-name-agnostic, taking any two series).
2. Calibration: reuses research/analyze_confidence.py::confidence_bucket_report()
   unmodified (column="momentum_return_pct") — the output CSV's direction_correct/
   actual_return_pct columns were named to match that function's existing contract
   for exactly this reason.

Usage:
    python -m research.analyze_momentum
    python -m research.analyze_momentum --input research/momentum_results.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from research.analyze_confidence import confidence_bucket_report
from research.analyze_smartscore import compute_rank_ic


def run(input_path: str) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze_momentum] {input_path} is empty — nothing to analyze.")
        return

    df = df.dropna(subset=["momentum_return_pct", "actual_return_pct"])
    print(f"[analyze_momentum] {len(df)} scored dates loaded from {input_path}\n")

    print("=== Rank-IC: trailing 12-1 momentum vs. forward return ===")
    ic = compute_rank_ic(df["momentum_return_pct"], df["actual_return_pct"])
    for k, v in ic.items():
        print(f"  {k}: {v}")
    print(
        "\n  Interpretation: near-zero or non-significant rank-IC means trailing "
        "momentum doesn't track forward return at all in this sample. Values above "
        "~0.03-0.05 are considered meaningful in equity return prediction; treat the "
        "sign and p-value as the first-order read, magnitude as secondary."
    )

    print("\n=== Momentum bucket report (quintiles) ===")
    buckets = confidence_bucket_report(df, n_buckets=5, column="momentum_return_pct")
    print(buckets.to_string(index=False))
    print(
        "\n  Interpretation: if win_rate_pct and mean_actual_return_pct climb roughly "
        "monotonically from the lowest to the highest momentum bucket, momentum is "
        "doing its job — a stock with stronger trailing performance is genuinely more "
        "likely to keep outperforming. If flat or non-monotonic (or inverted, "
        "consistent with mean-reversion instead of momentum), this factor isn't "
        "showing the expected edge in this sample either."
    )

    print("\n=== Overall directional accuracy ===")
    overall_win_rate = round(float(df["direction_correct"].mean()) * 100, 1)
    print(f"  n={len(df)}  directional_accuracy_pct={overall_win_rate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/momentum_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
