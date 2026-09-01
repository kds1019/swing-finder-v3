"""
Analyse the screener calibration dataset — Stage 2/3 of docs/strategy.md's plan.

Reads research/data/calibration_dataset.csv (built by build_calibration_dataset.py)
and, for every candidate feature, reports how the realised outcome varies across
its range. The point is to place each screener threshold where an edge actually
appears in the data, rather than where the single EMBJ reference trade happened to
sit — and to drop filters that carry no signal.

Outputs:
  - research/calibration_findings.md  (the tables + a short read)
  - prints the same tables to stderr

For each feature it shows, per decile / per fixed bin:
    n            row count in the bin
    hit_rate     fraction that reached the target (expired / stop count as miss)
    avg_R        mean R-multiple  (stop = -1.0 by construction)
    median_R     median R-multiple
    profit_factor  gross positive R / gross negative R  (>1 = net positive)

Then a walk-forward view: the same top-line metrics split by calendar year, so a
threshold that only worked in one regime is visible as such.

Usage:
    python -m research.analyze_calibration
    python -m research.analyze_calibration --dataset research/data/calibration_dataset.csv --weak-rr exclude
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "calibration_dataset.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "calibration_findings.md"

# Features to profile, with fixed human-meaningful bin edges where the shape matters
# more than equal-population deciles. None -> use deciles.
FEATURE_BINS: dict[str, list[float] | None] = {
    "price_vs_ema200_pct": [-25, -15, -10, -7, -4, -2, 0, 2, 4, 8, 15],
    "ema200_uptrend_pct": [0, 3, 5, 8, 12, 18, 25, 40, 100],
    "consolidation_range_pct": [0, 4, 6, 8, 10, 13, 16, 20, 30],
    "bounce_off_low_pct": [0, 1, 2, 3, 4, 6, 8, 12, 30],
    "price_vs_poc_pct": [-20, -8, -4, -2, 0, 2, 4, 8, 20],
    "price_vs_value_area_high_pct": [-25, -12, -8, -4, -2, 0, 3, 10],
    "atr_pct": [0, 1.5, 2, 2.5, 3, 4, 5, 7, 12],
    "rs_vs_spy_63d_pct": None,
    "rs_vs_spy_126d_pct": [-60, -30, -15, -5, 0, 5, 15, 30, 60, 150],
    "rs_vs_spy_252d_pct": None,
    "dist_52w_high_pct": [-60, -40, -30, -20, -15, -10, -7, -4, 0],
    "ema50_minus_ema200_pct": [-30, -15, -8, -4, 0, 4, 8, 15, 40],
}


def _metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0, "hit_rate": np.nan, "avg_R": np.nan, "median_R": np.nan, "profit_factor": np.nan}
    r = sub["r_multiple"].dropna()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n": n,
        "hit_rate": (sub["outcome"] == "target_hit").mean(),
        "avg_R": r.mean(),
        "median_R": r.median(),
        "profit_factor": (pos / neg) if neg > 0 else np.inf,
    }


def _fmt_row(label: str, m: dict) -> str:
    pf = "inf" if m["profit_factor"] == np.inf else f"{m['profit_factor']:.2f}"
    if m["n"] == 0:
        return f"| {label:>16} | {0:>6} |      - |      - |      - |     - |"
    return (f"| {label:>16} | {m['n']:>6} | {m['hit_rate']*100:5.1f}% | "
            f"{m['avg_R']:+6.3f} | {m['median_R']:+6.3f} | {pf:>5} |")


def _table(df: pd.DataFrame, feature: str, edges: list[float] | None) -> list[str]:
    s = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return [f"### {feature}", "", "_no data_", ""]
    sub = df.loc[s.index]

    if edges is None:
        q = np.unique(np.nanquantile(s, np.linspace(0, 1, 11)))
        edges = list(q)
    cats = pd.cut(sub[feature], bins=edges, include_lowest=True, duplicates="drop")

    lines = [f"### {feature}", "",
             "| bin | n | hit_rate | avg_R | median_R | PF |",
             "|---|---|---|---|---|---|"]
    for interval, g in sub.groupby(cats, observed=True):
        lines.append(_fmt_row(f"{interval.left:g}..{interval.right:g}", _metrics(g)))
    lines += ["", f"_overall:_ " + _fmt_row("all", _metrics(sub)).strip("|").strip(), ""]
    return lines


def _walk_forward(df: pd.DataFrame, mask_name: str, mask: pd.Series) -> list[str]:
    sub = df[mask]
    lines = [f"### walk-forward by year — {mask_name}  (n={len(sub)})", "",
             "| year | n | hit_rate | avg_R | median_R | PF |", "|---|---|---|---|---|---|"]
    yr = pd.to_datetime(sub["date"]).dt.year
    for y, g in sub.groupby(yr):
        lines.append(_fmt_row(str(y), _metrics(g)))
    lines += ["", _fmt_row("ALL", _metrics(sub)).replace("| ", "| **").replace(" |", "** |", 1), ""]
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description="Analyse the screener calibration dataset")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--weak-rr", choices=["keep", "exclude"], default="keep",
                   help="exclude rows the trade plan flagged weak_rr (mirrors the Decision Agent dropping them)")
    p.add_argument("--include-expired", choices=["yes", "no"], default="yes",
                   help="'no' drops expired_unresolved rows entirely (stricter hit-rate denominator)")
    args = p.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        sys.exit(f"{path} not found — run research.build_calibration_dataset first.")
    df = pd.read_csv(path, parse_dates=["date"])
    n0 = len(df)

    if args.weak_rr == "exclude":
        df = df[~df["weak_rr"].astype(bool)]
    if args.include_expired == "no":
        df = df[df["outcome"] != "expired_unresolved"]
    df = df.reset_index(drop=True)

    out: list[str] = [
        "# Screener calibration findings", "",
        f"- dataset: `{path}`  ({n0:,} rows, {len(df):,} after filters)",
        f"- weak_rr rows: **{args.weak_rr}**   |   expired_unresolved: **{args.include_expired}**",
        f"- tickers: {df['ticker'].nunique()}   |   "
        f"date range: {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}",
        "- stop_hit is exactly -1.0 R by construction; avg_R > 0 and PF > 1 mean a net edge in that slice.",
        "- Alpaca IEX history caveat: ~one cycle only, survivorship-biased to today's universe.",
        "",
        "## Top line", "",
        "| slice | n | hit_rate | avg_R | median_R | PF |", "|---|---|---|---|---|---|",
        _fmt_row("all wide-net", _metrics(df)),
        _fmt_row("current thresholds", _metrics(df[df["detected"].astype(bool)])),
        "",
        "## Per-feature", "",
        "Read each feature for a *monotonic* or *plateau* relationship between the bin and "
        "avg_R / PF. A single spiking bin is noise; a run of bins that clears PF>1 is where "
        "the gate belongs. Compare against where the current cutoff sits.", "",
    ]

    for feature, edges in FEATURE_BINS.items():
        if feature in df.columns:
            out += _table(df, feature, edges)

    # boolean feature
    if "ema50_gt_ema200" in df.columns:
        out += ["### ema50_gt_ema200 (boolean)", "",
                "| value | n | hit_rate | avg_R | median_R | PF |", "|---|---|---|---|---|---|"]
        for val, g in df.groupby(df["ema50_gt_ema200"].astype(bool)):
            out.append(_fmt_row(str(val), _metrics(g)))
        out.append("")

    out += ["## Walk-forward", ""]
    out += _walk_forward(df, "all wide-net", pd.Series(True, index=df.index))
    out += _walk_forward(df, "current thresholds", df["detected"].astype(bool))

    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out), file=sys.stderr)
    print(f"\n[analyze] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
