"""
Phase 2 of docs/ml-edge-confidence-research.md: analysis over the CSV produced
by research/walk_forward_backtest.py. Answers the question neither the live
pipeline nor the research doc could answer without data: does the ensemble's
`confidence` output actually track forward-return accuracy, and by how much?

Three views, each a hand-rolled version of what a heavier framework would give
you (see docs/ml-edge-confidence-research.md for why those weren't imported
wholesale):
  - IC / rank-IC (Qlib's core evaluation metric) — does predicted return
    magnitude correlate with realized return magnitude, not just direction.
  - Calibration by confidence decile (ml4t-style reliability check) — does a
    higher-confidence bucket actually hit direction more often / more
    positively than a lower one.
  - Confidence-bucket backtest (vectorbt-style) — mean/median forward return
    and win rate per confidence bucket.

Usage:
    python -m research.analyze_confidence
    python -m research.analyze_confidence --input research/walk_forward_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(df: pd.DataFrame) -> dict:
    pearson_ic, pearson_p = stats.pearsonr(df["predicted_return_pct"], df["actual_return_pct"])
    spearman_ic, spearman_p = stats.spearmanr(df["predicted_return_pct"], df["actual_return_pct"])
    return {
        "n": len(df),
        "ic_pearson": round(float(pearson_ic), 4),
        "ic_pearson_pvalue": round(float(pearson_p), 4),
        "rank_ic_spearman": round(float(spearman_ic), 4),
        "rank_ic_spearman_pvalue": round(float(spearman_p), 4),
    }


def confidence_bucket_report(df: pd.DataFrame, n_buckets: int = 5) -> pd.DataFrame:
    """Deciles by default n_buckets=5 (quintiles) — walk-forward sample sizes rarely
    support true deciles without individual buckets getting too small to trust."""
    labeled = df.copy()
    try:
        labeled["bucket"] = pd.qcut(labeled["confidence"], n_buckets, duplicates="drop")
    except ValueError:
        # Not enough distinct confidence values to form n_buckets groups.
        labeled["bucket"] = pd.cut(labeled["confidence"], min(n_buckets, labeled["confidence"].nunique()))

    report = labeled.groupby("bucket", observed=True).agg(
        n=("confidence", "size"),
        confidence_min=("confidence", "min"),
        confidence_max=("confidence", "max"),
        win_rate_pct=("direction_correct", lambda s: round(float(s.mean()) * 100, 1)),
        mean_actual_return_pct=("actual_return_pct", lambda s: round(float(s.mean()), 3)),
        median_actual_return_pct=("actual_return_pct", lambda s: round(float(s.median()), 3)),
    ).reset_index()
    return report


def run(input_path: str) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze] {input_path} is empty — nothing to analyze.")
        return

    df = df.dropna(subset=["predicted_return_pct", "actual_return_pct", "confidence"])
    print(f"[analyze] {len(df)} scored walk-forward predictions loaded from {input_path}\n")

    print("=== IC / rank-IC (predicted vs. actual return magnitude) ===")
    ic = compute_ic(df)
    for k, v in ic.items():
        print(f"  {k}: {v}")
    print(
        "\n  Interpretation: IC/rank-IC near 0 means predicted return magnitude carries no "
        "information about actual return magnitude, even if directional accuracy alone looks "
        "OK. Values above ~0.03-0.05 are considered meaningful in equity return prediction "
        "(daily-frequency signals are extremely noisy); this is a 5-day-ahead swing signal so "
        "there's no established benchmark to compare against — treat the sign and p-value as "
        "the first-order read (is there any real correlation at all, statistically), and the "
        "magnitude as secondary."
    )

    print("\n=== Confidence bucket report (quintiles) ===")
    buckets = confidence_bucket_report(df)
    print(buckets.to_string(index=False))
    print(
        "\n  Interpretation: if win_rate_pct and mean_actual_return_pct both increase "
        "monotonically from the lowest to the highest confidence bucket, confidence is doing "
        "its job. If they're flat or non-monotonic, the current confidence formula "
        "(core/ml_forecast.py's dir_conf * r2_adj) isn't actually separating good calls from "
        "bad ones, and the scaled SmartScore adjustment (ML_EDGE_CONFIDENCE_SATURATION) and "
        "any future meta-labeling work should be revisited rather than assumed correct."
    )

    print("\n=== Overall directional accuracy (for comparison to ml_track_record) ===")
    overall_win_rate = round(float(df["direction_correct"].mean()) * 100, 1)
    print(f"  n={len(df)}  directional_accuracy_pct={overall_win_rate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/walk_forward_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
