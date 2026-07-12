"""
Analysis for research/rules_based_walk_forward.py's output — does the base rules-based
system (core.smartscore + core.trade_plan, no ML) have a real edge on its own?

Two checks, both new relative to every prior analysis script this session (which all
evaluated an ML layer, never the base system underneath it):

1. Expectancy: is the realized average R-multiple per trade significantly greater than
   zero? Each trade's R-multiple is exactly +rr_ratio (target_hit) or -1 (stop_hit) by
   construction — resolve_trade_plan_outcome() returns the exact stop/target price, not a
   partial fill — so this is a direct, per-trade expected-value check, not a proxy.
2. Calibration: does SmartScore's own ranking separate good setups from bad ones? Reuses
   research/analyze_confidence.py::confidence_bucket_report() unmodified (column="smartscore")
   — the output CSV's direction_correct/actual_return_pct columns were named to match that
   function's existing contract for exactly this reason.

Usage:
    python -m research.analyze_rules_based
    python -m research.analyze_rules_based --input research/rules_based_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from research.analyze_confidence import confidence_bucket_report


def compute_expectancy(df: pd.DataFrame) -> dict:
    """Per-trade R-multiple: +rr_ratio if target was hit first, -1 if stop was hit first
    (by construction, since resolve_trade_plan_outcome resolves to the exact stop/target
    price, never a partial fill). Mean R-multiple is the system's expectancy in the
    standard trading sense — positive means the average trade makes money net of losers,
    independent of position sizing."""
    r_multiple = np.where(df["direction_correct"], df["rr_ratio"], -1.0)
    mean_r, std_r = float(np.mean(r_multiple)), float(np.std(r_multiple, ddof=1))
    t_stat, p_value = stats.ttest_1samp(r_multiple, popmean=0.0)
    return {
        "n": len(df),
        "win_rate_pct": round(float(df["direction_correct"].mean()) * 100, 1),
        "mean_rr_ratio": round(float(df["rr_ratio"].mean()), 2),
        "mean_r_multiple": round(mean_r, 4),
        "std_r_multiple": round(std_r, 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
    }


def run(input_path: str) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        print(f"[analyze_rules_based] {input_path} is empty — nothing to analyze.")
        return

    df = df.dropna(subset=["smartscore", "direction_correct", "actual_return_pct", "rr_ratio"])
    print(f"[analyze_rules_based] {len(df)} scored trade plans loaded from {input_path}\n")

    print("=== Expectancy: is the base system's average trade profitable? ===")
    exp = compute_expectancy(df)
    for k, v in exp.items():
        print(f"  {k}: {v}")
    print(
        "\n  Interpretation: mean_r_multiple is the average number of risk-units (R) won or "
        "lost per trade, independent of position sizing — positive and statistically "
        "significant (p_value < 0.05) means the base entry/stop/target logic has a real "
        "edge on its own, before any ML overlay. Zero or negative means the system's win "
        "rate doesn't clear the bar its own R:R ratios require — the near-33% expectation "
        "documented in settings.min_risk_reward's 3:1 floor requires roughly a 25%+ win "
        "rate just to break even; compare win_rate_pct against that intuition directly."
    )

    print("\n=== SmartScore bucket report (quintiles) — does ranking separate good from bad? ===")
    buckets = confidence_bucket_report(df, n_buckets=5, column="smartscore")
    print(buckets.to_string(index=False))
    print(
        "\n  Interpretation: if win_rate_pct and mean_actual_return_pct climb roughly "
        "monotonically from the lowest to the highest SmartScore bucket, the score is "
        "doing its job — a higher-ranked pick is genuinely more likely to work. If flat "
        "or non-monotonic, SmartScore isn't actually separating good setups from bad "
        "ones, independent of whatever its expectancy looks like in aggregate."
    )

    print("\n=== By setup type (Breakout vs Pullback vs near-miss-only) ===")
    by_setup = df.copy()
    by_setup["setup_label"] = by_setup["setup"].fillna("near_miss_only")
    setup_report = by_setup.groupby("setup_label").apply(
        lambda g: pd.Series({
            "n": len(g),
            "win_rate_pct": round(float(g["direction_correct"].mean()) * 100, 1),
            "mean_actual_return_pct": round(float(g["actual_return_pct"].mean()), 3),
        }),
        include_groups=False,
    )
    print(setup_report.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/rules_based_results.csv")
    args = parser.parse_args()
    run(args.input)


if __name__ == "__main__":
    main()
