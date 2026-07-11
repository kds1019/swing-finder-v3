"""
Analysis for research/triple_barrier_walk_forward.py's output — the classification
analogue of research/analyze_confidence.py. That script's compute_ic()/run() are wired to
the regression CSV's columns (predicted_return_pct, confidence) and don't apply here (this
CSV has a probability, p_target, against a binary label, not two continuous return series),
but confidence_bucket_report() was already generalized to take an arbitrary probability
column and is reused as-is — same calibration-by-bucket check, new column name.

Usage:
    python -m research.analyze_triple_barrier
    python -m research.analyze_triple_barrier --input research/triple_barrier_results.csv
"""

from __future__ import annotations

import argparse

from scipy import stats
from sklearn.metrics import roc_auc_score

from research.analyze_confidence import confidence_bucket_report


def run(input_path: str) -> None:
    import pandas as pd

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze_triple_barrier] {input_path} is empty — nothing to analyze.")
        return

    df = df.dropna(subset=["p_target", "direction_correct", "actual_return_pct"])
    print(f"[analyze_triple_barrier] {len(df)} held-out triple-barrier predictions loaded "
          f"from {input_path}\n")

    print("=== AUC and point-biserial correlation (p_target vs. hit-target-first) ===")
    auc = roc_auc_score(df["direction_correct"], df["p_target"])
    corr, corr_p = stats.pointbiserialr(df["direction_correct"], df["p_target"])
    print(f"  n={len(df)}")
    print(f"  auc={auc:.4f}  (0.5 = no better than chance, 1.0 = perfect ranking)")
    print(f"  point_biserial_r={corr:.4f} p={corr_p:.4f}")

    print("\n=== p_target bucket report (quintiles) — calibration check ===")
    buckets = confidence_bucket_report(df, n_buckets=5, column="p_target")
    print(buckets.to_string(index=False))
    print(
        "\n  Interpretation: win_rate_pct here IS P(target hit first) realized in each "
        "bucket — if it climbs roughly in step with the bucket's own p_target range (e.g. "
        "the ~70-80% p_target bucket actually hits target ~70-80% of the time), the "
        "classifier is usably calibrated, the bar docs/ml-edge-confidence-research.md sets "
        "before this could inform expected-value ranking or position sizing. If it's flat "
        "or non-monotonic, this reframe hasn't found a real edge either, same conclusion as "
        "the regression ensemble after the stale-features fix — treat p_target as unproven "
        "until this check passes."
    )

    print("\n=== Overall hit rate (for comparison to pick_outcomes.csv win_rate_pct) ===")
    overall_hit_rate = round(float(df["direction_correct"].mean()) * 100, 1)
    print(f"  n={len(df)}  target_hit_first_pct={overall_hit_rate}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/triple_barrier_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
