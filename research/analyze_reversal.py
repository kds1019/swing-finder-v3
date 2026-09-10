"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Answers the V3 audit's core questions from research/data/reversal_features.csv
(built by build_reversal_features.py):

  3/5. Which "has it stopped falling?" / decline-character variables actually
       separate DEEP PULLBACK + REVERSAL from DEEP PULLBACK + STILL FALLING.
  4.   The confirmation trade-off: expected-R cost of waiting for stabilisation
       vs entering at the low.
  6.   Gate ablation: which current technical gates carry independent value.
  10.  Everything is reported TRAIN (2021-2024) vs TEST (2025-2026), with 2022
       broken out, and with low sample sizes shown, not hidden.

Primary outcome = realised R under the LIVE exit (tr_2r_g1: hold initial stop to
+2R, then trail peak-1R). Falling-knife proxy = fixed stop hit within 5 bars.

Usage:  python -m research.analyze_reversal
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SRC = DATA / "reversal_features.csv"
OUT = Path(__file__).resolve().parent / "reversal_findings.md"

R_COL = "tr_2r_g1"          # realised R under the live trailing exit
TRAIN_END = "2025-01-01"    # 2021-2024 train, 2025-2026 test

# current live gates (core.pullback_reversal)
GATES = {
    "G1_ema200_up>=5":       lambda d: d.ema200_uptrend_pct >= 5.0,
    "G2_price_vs_ema200":     lambda d: d.price_vs_ema200_pct.between(-20.0, 3.0),
    "G3_range<=20":          lambda d: d.consolidation_range_pct <= 20.0,
    "G4_vah<=-4":            lambda d: d.price_vs_value_area_high_pct <= -4.0,
}


def load() -> pd.DataFrame:
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d = d[~d["weak_rr"].astype(bool)].copy()          # mirrors the Decision Agent dropping weak-RR
    d["knife5"] = ((d.outcome == "stop_hit") & (d.bars_to_resolution <= 5)).astype(int)
    d["win"] = (d[R_COL] > 0).astype(int)
    d["split"] = np.where(d.date < TRAIN_END, "train", "test")
    return d


def _mcl(r: pd.Series) -> int:
    """max consecutive losers (R<=0) over the date-ordered sequence"""
    loss = (r.values <= 0).astype(int)
    best = cur = 0
    for x in loss:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def metrics(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return dict(n=0)
    r = d[R_COL].replace([np.inf, -np.inf], np.nan).dropna()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    ds = d.sort_values("date")
    return dict(
        n=len(d),
        win=round(100 * d.win.mean(), 1),
        avgR=round(r.mean(), 3),
        medR=round(r.median(), 2),
        PF=round(pos / neg, 2) if neg > 0 else np.inf,
        MAE=round(d.mae_r.mean(), 2),
        knife5=round(100 * d.knife5.mean(), 1),
        mcl=_mcl(ds[R_COL]),
    )


def _dedupe(d: pd.DataFrame, gap: int = 10) -> pd.DataFrame:
    """~one signal per pullback episode per ticker: drop rows within `gap` trading
    days of an already-kept row for the same ticker (rows are ~daily samples of the
    same setup, heavily autocorrelated)."""
    keep = []
    last: dict[str, pd.Timestamp] = {}
    for idx, row in d.sort_values(["ticker", "date"]).iterrows():
        t, dt = row["ticker"], row["date"]
        if t not in last or (dt - last[t]).days >= gap:
            keep.append(idx)
            last[t] = dt
    return d.loc[keep]


def fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "n=0"
    return (f"n={m['n']:>6}  win={m['win']:>5}%  avgR={m['avgR']:>+6.3f}  medR={m['medR']:>+5.2f}  "
            f"PF={m['PF']:>5}  MAE={m['MAE']:>+5.2f}  knife5={m['knife5']:>5}%  maxConsecLoss={m['mcl']}")


def section(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def bin_table(d: pd.DataFrame, col: str, edges, lines: list[str], on_test=True) -> None:
    lines.append(f"### {col}")
    lines.append("")
    s = d[col].replace([np.inf, -np.inf], np.nan)
    cats = pd.cut(s, bins=edges, include_lowest=True, duplicates="drop")
    for split in (["train", "test"] if on_test else ["train"]):
        sub = d[d.split == split]
        lines.append(f"_{split}_")
        for iv, g in sub.groupby(cats, observed=True):
            m = metrics(g)
            lines.append(f"  {str(iv):>16}  {fmt(m)}")
        lines.append("")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} not found — run research.build_reversal_features first")
    d = load()
    net = d                                    # wide net (rising-200EMA pullback region)
    det = d[d["detected"].astype(bool)]        # passes current live gates
    L: list[str] = ["# Reversal / falling-knife findings (RESEARCH)", "",
                    f"- source: {SRC.name}   rows(after weak_rr drop): {len(d):,}   R metric: {R_COL} (live trailing exit)",
                    f"- train = date < {TRAIN_END} (2021-2024) | test = 2025-2026 | knife5 = fixed stop hit <=5 bars",
                    "- MAE from the fixed-stop path; avgR/win/PF/maxConsecLoss from the trailing exit",
                    "- rows are ~daily samples of the same setups (autocorrelated); episode-deduped headline below"]

    L += section("0. Baselines (train / test / 2022) — wide net vs current gates")
    for name, sub in [("wide net", net), ("current gates", det)]:
        L.append(f"**{name}**")
        for split in ["train", "test"]:
            L.append(f"  {split:5}  {fmt(metrics(sub[sub.split == split]))}")
        L.append(f"  2022   {fmt(metrics(sub[sub.date.dt.year == 2022]))}")
        dd = _dedupe(sub)
        L.append(f"  train (episode-deduped)  {fmt(metrics(dd[dd.split=='train']))}")
        L.append(f"  test  (episode-deduped)  {fmt(metrics(dd[dd.split=='test']))}")
        L.append("")

    # ---------- 3/5. feature discovery on TRAIN, confirm on TEST ----------
    L += section("1. Falling-knife discriminators — univariate, current-gate subset")
    L.append("Spearman corr vs realised R and vs knife5, TRAIN only (current-gate rows):")
    L.append("")
    feat_cols = [c for c in d.columns if c.startswith("rev_")]
    tr = det[det.split == "train"]
    rows = []
    for c in feat_cols:
        x = tr[c].replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 500:
            continue
        cr = x.corr(tr[R_COL], method="spearman")
        ck = x.corr(tr["knife5"], method="spearman")
        rows.append((c, cr, ck))
    rows.sort(key=lambda t: -abs(t[1]))
    L.append(f"  {'feature':<32}{'rho(R)':>9}{'rho(knife5)':>13}")
    for c, cr, ck in rows:
        L.append(f"  {c:<32}{cr:>+9.3f}{ck:>+13.3f}")
    L.append("")

    # bin tables for the most-promising, train + test
    show = [
        ("rev_days_since_low_20", [-0.1, 0.5, 1.5, 3.5, 6.5, 10.5, 20.1]),
        ("rev_higher_low_pct", [-30, -0.01, 1, 3, 6, 12, 60]),
        ("rev_close_vs_ema20_pct", [-40, -12, -8, -4, -1, 1, 4, 20]),
        ("rev_days_below_ema20", [-0.1, 0.5, 3.5, 8.5, 15.5, 40, 300]),
        ("rev_ema20_slope_5d_pct", [-30, -4, -2, -0.5, 0.5, 2, 20]),
        ("rev_reclaimed_ema20", [-0.1, 0.5, 1.1]),
        ("rev_max_1d_drop_pct", [-60, -15, -10, -7, -5, -3, 0]),
        ("rev_max_3d_drop_pct", [-70, -20, -14, -10, -7, 0]),
        ("rev_worst_gap_pct", [-60, -12, -7, -4, -2, 0, 20]),
        ("rev_decline_in_atr", [0, 2, 3.5, 5, 7, 10, 40]),
        ("rev_waterfall_ratio", [0, 0.3, 0.45, 0.6, 0.8, 3]),
        ("rev_down_days_10", [-0.1, 3.5, 5.5, 6.5, 7.5, 10.1]),
        ("rev_last_5d_return_pct", [-40, -8, -4, -1, 1, 4, 40]),
        ("rev_down_up_vol_ratio_12", [0, 0.7, 1.0, 1.3, 1.8, 10]),
        ("rev_vol_at_low_rel", [0, 0.8, 1.2, 1.8, 2.5, 20]),
        ("rev_up_vol_expansion_5", [0, 0.6, 0.9, 1.2, 1.6, 10]),
        ("rev_rsi14", [0, 25, 32, 38, 45, 55, 100]),
        ("rev_rsi_min_10", [0, 20, 27, 33, 40, 100]),
        ("rev_rsi_turn_up", [-0.1, 0.5, 1.1]),
        ("rev_rsi_bull_div", [-0.1, 0.5, 1.1]),
        ("rev_macd_hist_up", [-0.1, 0.5, 1.1]),
        ("rev_dist_to_swing_low_20_pct", [-0.1, 1, 3, 6, 10, 60]),
        ("rev_ema50_slope_20d_pct", [-30, -1, 0, 1, 3, 6, 40]),
        ("rev_pct_bars_above_ema50_126", [0, 40, 55, 70, 85, 100]),
        ("rev_trend_linearity_120", [-1.01, 0, 0.4, 0.7, 0.85, 1.01]),
        ("price_vs_ema200_pct", [-40, -25, -20, -15, -10, -5, 0, 3, 8, 15]),
        ("dist_52w_high_pct", [-70, -40, -30, -20, -12, -7, -4, 0]),
    ]
    L += section("2. Bin tables (train + test) — most-informative features")
    for col, edges in show:
        if col in d.columns:
            bin_table(det, col, edges, L)

    # ---------- 4. confirmation trade-off ----------
    L += section("3. Confirmation trade-off — cost of waiting for stabilisation")
    L.append("Compare current-gate rows split by whether stabilisation is ALREADY visible at the signal.")
    L.append("`stabilising` = days_since_low_20>=3 AND higher_low_pct>0 AND close_vs_ema20_pct>-6 AND last_5d_return_pct>-4")
    L.append("`still_falling` = the complement. Entry-price-given-up = how far price already sits above the 20d low.")
    L.append("")
    stab = ((det.rev_days_since_low_20 >= 3) & (det.rev_higher_low_pct > 0)
            & (det.rev_close_vs_ema20_pct > -6) & (det.rev_last_5d_return_pct > -4))
    for split in ["train", "test"]:
        sub = det[det.split == split]
        sm = stab.loc[sub.index]
        L.append(f"_{split}_")
        L.append(f"  stabilising     {fmt(metrics(sub[sm]))}   dist_above_20d_low(med)={sub[sm].rev_dist_to_swing_low_20_pct.median():.1f}%")
        L.append(f"  still_falling    {fmt(metrics(sub[~sm]))}   dist_above_20d_low(med)={sub[~sm].rev_dist_to_swing_low_20_pct.median():.1f}%")
        L.append("")
    # tighter "good confirmation" — stabilising but still near the low
    good = stab & (det.rev_dist_to_swing_low_20_pct <= 6)
    late = stab & (det.rev_dist_to_swing_low_20_pct > 6)
    L.append("Within `stabilising`: near the low (<=6% above 20d low) vs already bounced (>6%):")
    for split in ["train", "test"]:
        sub = det[det.split == split]
        L.append(f"_{split}_")
        L.append(f"  stabilising & near low   {fmt(metrics(sub[good.loc[sub.index]]))}")
        L.append(f"  stabilising & bounced    {fmt(metrics(sub[late.loc[sub.index]]))}")
        L.append("")

    # ---------- 6. gate ablation ----------
    L += section("4. Gate ablation — independent value of each current gate")
    base = net
    L.append("Applied to the wide net. Each line = that gate set, train then test.")
    L.append("")
    def show_mask(label, mask):
        L.append(f"**{label}**")
        for split in ["train", "test"]:
            sub = base[(base.split == split) & mask.loc[base.index]]
            L.append(f"  {split:5} {fmt(metrics(sub))}")
        L.append("")
    all_true = pd.Series(True, index=base.index)
    show_mask("wide net (no gates)", all_true)
    gmask = {k: f(base) for k, f in GATES.items()}
    for k, m in gmask.items():
        show_mask(f"only {k}", m)
    allg = all_true.copy()
    for m in gmask.values():
        allg &= m
    show_mask("ALL current gates", allg)
    for k in GATES:
        loo = all_true.copy()
        for kk, m in gmask.items():
            if kk != k:
                loo &= m
        show_mask(f"ALL except {k}", loo)
    # variants
    L.append("**G2 lower-bound variants (with G1,G3,G4 held):**")
    g134 = gmask["G1_ema200_up>=5"] & gmask["G3_range<=20"] & gmask["G4_vah<=-4"]
    for lo in (-30, -25, -20, -15):
        for hi in (3, 8):
            m = g134 & base.price_vs_ema200_pct.between(lo, hi)
            for split in ["train", "test"]:
                sub = base[(base.split == split) & m.loc[base.index]]
                L.append(f"  [{lo:>4},{hi:>2}] {split:5} {fmt(metrics(sub))}")
            L.append("")
    L.append("**G1 threshold variants (with G2[-20,3],G3,G4 held):**")
    g234 = base.price_vs_ema200_pct.between(-20, 3) & gmask["G3_range<=20"] & gmask["G4_vah<=-4"]
    for thr in (0, 3, 5, 8, 12):
        m = g234 & (base.ema200_uptrend_pct >= thr)
        for split in ["train", "test"]:
            sub = base[(base.split == split) & m.loc[base.index]]
            L.append(f"  >={thr:>2} {split:5} {fmt(metrics(sub))}")
        L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"[analyze] wrote {OUT}  ({len(L)} lines)", file=sys.stderr)
    print("\n".join(L))


if __name__ == "__main__":
    main()
