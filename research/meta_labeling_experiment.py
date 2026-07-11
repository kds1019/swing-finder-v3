"""
Meta-labeling (Lopez de Prado): train a secondary classifier whose only job is to
predict whether the primary ensemble's directional call will be correct, using
summary statistics about that call (confidence, per-model R^2, RF/GB agreement) as
features — rather than trying to predict the return itself. Its output probability
is a principled replacement for the ad hoc `dir_conf * r2_adj` confidence formula in
core/ml_forecast.py, which every walk-forward run so far has shown produces flat-to-
inverted confidence buckets (see docs/ml-edge-confidence-research.md) regardless of
what's fed into the *primary* model.

This was explicitly deferred earlier in this project for lack of data — that blocker
is gone now that research/walk_forward_backtest.py can generate a few thousand
(primary-call summary, actual outcome) rows in one run, instead of waiting months for
the live ml_predictions.csv log to grow one row at a time.

Reads research/walk_forward_results.csv (already produced by walk_forward_backtest.py
— this script trains on it directly, no new data fetch, no network access needed).

Usage:
    python -m research.meta_labeling_experiment
    python -m research.meta_labeling_experiment --input research/walk_forward_results.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from research.analyze_confidence import confidence_bucket_report
from research.pooled_model_experiment import time_based_split

META_FEATURES = ["rf_confidence", "gb_confidence", "rf_r2", "gb_r2", "agreement_pct", "confidence"]


def prepare_meta_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=META_FEATURES + ["direction_correct", "actual_return_pct"]).copy()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["direction_correct"] = df["direction_correct"].astype(bool).astype(int)
    return df


def evaluate_meta_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    # AUC=0.5 / no discrimination baseline: a Brier score at least as good as always
    # predicting the base rate tells us the model isn't actively worse than guessing.
    base_rate = y_true.mean()
    brier_base_rate = brier_score_loss(y_true, np.full_like(y_true, base_rate, dtype=float))
    print(f"\n=== {name} — held-out test period ===")
    print(f"  n={len(y_true)}  base_rate={base_rate:.3f}")
    print(f"  AUC={auc:.4f}  (0.5 = no better than guessing)")
    print(f"  Brier score={brier:.4f}  (base-rate-only baseline={brier_base_rate:.4f}, lower is better)")


def run(input_path: str, train_frac: float) -> None:
    raw = pd.read_csv(input_path)
    if raw.empty:
        print(f"[meta] {input_path} is empty — nothing to train on.")
        return

    df = prepare_meta_dataset(raw)
    print(f"[meta] {len(df)} usable rows (of {len(raw)} total) after dropping rows missing "
          f"any meta-feature")

    train, test = time_based_split(df, train_frac)
    print(f"[meta] time-based split: {len(train)} train rows (< {train['as_of_date'].max().date()}), "
          f"{len(test)} test rows (>= {test['as_of_date'].min().date()})")

    X_train, y_train = train[META_FEATURES].values, train["direction_correct"].values
    X_test, y_test = test[META_FEATURES].values, test["direction_correct"].values

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    evaluate_meta_model("Logistic Regression meta-model", y_test, lr_prob)
    print("  Coefficients:", dict(zip(META_FEATURES, np.round(lr.coef_[0], 4))))

    # Shallow + high min_samples_leaf deliberately, matching this project's existing primary
    # models' regularization philosophy — 6 features and a few thousand rows is not much
    # margin, and meta-labeling's whole premise (a small, disciplined secondary model) breaks
    # if the meta-model itself overfits the noise it's supposed to be filtering out.
    rf = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=30, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    evaluate_meta_model("Random Forest meta-model", y_test, rf_prob)
    print("  Feature importances:", dict(zip(META_FEATURES, np.round(rf.feature_importances_, 4))))

    print("\n=== Calibration check: meta-model probability vs. actual outcome, same test period ===")
    test_with_probs = test.copy()
    test_with_probs["lr_prob"] = lr_prob
    test_with_probs["rf_prob"] = rf_prob

    print("\n-- Original confidence formula, on this test period only (for a fair comparison) --")
    print(confidence_bucket_report(test_with_probs, column="confidence").to_string(index=False))

    print("\n-- Logistic Regression meta-model probability --")
    print(confidence_bucket_report(test_with_probs, column="lr_prob").to_string(index=False))

    print("\n-- Random Forest meta-model probability --")
    print(confidence_bucket_report(test_with_probs, column="rf_prob").to_string(index=False))

    print(
        "\n  Interpretation: compare win_rate_pct across buckets for each of the three sections "
        "above, all computed on the identical held-out rows. If a meta-model's win_rate_pct "
        "increases monotonically from lowest to highest bucket where the original confidence "
        "formula didn't, that meta-model is a real calibration improvement worth wiring into "
        "core/ml_forecast.py in place of the current dir_conf * r2_adj formula. If none of them "
        "show a clean monotonic trend, the underlying primary-model signal may just be too weak "
        "for any confidence formula to meaningfully separate good calls from bad ones on this "
        "sample size — not a meta-labeling implementation problem."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="research/walk_forward_results.csv")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()
    run(args.input, args.train_frac)


if __name__ == "__main__":
    main()
