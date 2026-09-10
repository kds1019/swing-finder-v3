"""
RESEARCH / AUDIT ONLY. Supplementary cuts for the V3 reversal audit:
  - decline character vs the falling-knife outcome (item 5)
  - candidate-volume impact of a stabilisation filter (items 8/9/13)
  - can any small interpretable model beat the knife5 base rate (item 3)
  - depth x days-since-low interaction
  - "abnormal single-day collapse" (earnings-gap proxy)

Usage:  python -m research.analyze_reversal_supp
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

SRC = Path(__file__).resolve().parent / "data" / "reversal_features.csv"
OUT = Path(__file__).resolve().parent / "reversal_findings_supp.md"
R = "tr_2r_g1"
TRAIN_END = "2025-01-01"


def m(d):
    if len(d) == 0:
        return "n=0"
    r = d[R].replace([np.inf, -np.inf], np.nan).dropna()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return (f"n={len(d):>6} win={100*(d[R]>0).mean():>4.0f}% avgR={r.mean():>+5.2f} "
            f"PF={pos/neg if neg>0 else 9.99:>4.2f} MAE={d.mae_r.mean():>+5.2f} "
            f"knife5={100*d.knife5.mean():>4.0f}%")


def main():
    d = pd.read_csv(SRC, parse_dates=["date"], low_memory=False)
    d = d[~d.weak_rr.astype(bool)].copy()
    d["knife5"] = ((d.outcome == "stop_hit") & (d.bars_to_resolution <= 5)).astype(int)
    d["split"] = np.where(d.date < TRAIN_END, "train", "test")
    det = d[d.detected.astype(bool)]
    L = ["# Reversal audit — supplementary (RESEARCH)", ""]

    L += ["## A. Decline character vs falling knife (current-gate set, train/test)", ""]
    for col, edges in [
        ("rev_max_1d_drop_pct", [-60, -12, -8, -6, -4, 0]),
        ("rev_worst_gap_pct", [-60, -10, -6, -3, -1, 0.01]),
        ("rev_waterfall_ratio", [0, 0.4, 0.55, 0.7, 0.85, 3]),
        ("rev_decline_in_atr", [0, 2.5, 4, 5.5, 8, 40]),
    ]:
        L.append(f"### {col}")
        for sp in ["train", "test"]:
            L.append(f"_{sp}_")
            s = det[det.split == sp]
            for iv, g in s.groupby(pd.cut(s[col], edges, include_lowest=True), observed=True):
                L.append(f"  {str(iv):>16} {m(g)}")
        L.append("")

    L += ["## B. 'Abnormal single-day collapse' — worst 1-day drop <= -12% anywhere in the pullback", ""]
    for sp in ["train", "test"]:
        s = det[det.split == sp]
        coll = s.rev_max_1d_drop_pct <= -12
        L.append(f"_{sp}_  collapse:   {m(s[coll])}")
        L.append(f"_{sp}_  no-collapse:{m(s[~coll])}")
    L.append("")

    L += ["## C. Stabilisation filter — candidate-volume impact", ""]
    stab = ((det.rev_days_since_low_20 >= 3) & (det.rev_higher_low_pct > 0)
            & (det.rev_close_vs_ema20_pct > -6) & (det.rev_last_5d_return_pct > -4))
    det2 = det.assign(stab=stab)
    # signals per calendar month = rough proxy for signals per run
    per_mo = det2.groupby(det2.date.dt.to_period("M")).agg(all=("ticker", "size"),
                                                           stab=("stab", "sum"))
    per_mo["ratio"] = per_mo.stab / per_mo["all"]
    L.append(f"- current-gate signals/month: median {per_mo['all'].median():.0f}  "
             f"(p25 {per_mo['all'].quantile(.25):.0f}, p75 {per_mo['all'].quantile(.75):.0f})")
    L.append(f"- stabilising signals/month: median {per_mo['stab'].median():.0f}  "
             f"(p25 {per_mo['stab'].quantile(.25):.0f}, p75 {per_mo['stab'].quantile(.75):.0f})")
    L.append(f"- stabilising share: mean {per_mo['ratio'].mean():.0%}")
    L.append(f"- weeks with 0 stabilising signals: {(per_mo['stab']==0).mean():.0%} of months")
    L.append("")
    # trimmed-gate (drop G1 slope + G3 range) volume, for comparison
    trim = d[d.price_vs_ema200_pct.between(-20, 3)
             & d.price_vs_value_area_high_pct.notna()
             & (d.price_vs_value_area_high_pct <= -4)]
    tstab = ((trim.rev_days_since_low_20 >= 3) & (trim.rev_higher_low_pct > 0)
             & (trim.rev_close_vs_ema20_pct > -6) & (trim.rev_last_5d_return_pct > -4))
    tpm = trim.assign(s=tstab).groupby(trim.date.dt.to_period("M")).agg(all=("ticker", "size"), s=("s", "sum"))
    L.append(f"- trimmed-gate (no G1/G3) signals/month: median all {tpm['all'].median():.0f}, "
             f"stabilising {tpm['s'].median():.0f}")
    L.append("")

    L += ["## D. Depth x days-since-low (current-gate, test set)", "",
          "avgR / knife5 by (price_vs_ema200 bin) x (days_since_low bin)", ""]
    s = det[det.split == "test"].copy()
    s["db"] = pd.cut(s.price_vs_ema200_pct, [-25, -10, -5, 0, 3], include_lowest=True)
    s["lb"] = pd.cut(s.rev_days_since_low_20, [-.1, 0.5, 2.5, 5.5, 20.1], include_lowest=True)
    piv_r = s.pivot_table(R, "db", "lb", "mean", observed=True)
    piv_k = s.pivot_table("knife5", "db", "lb", "mean", observed=True)
    L.append("avgR:"); L.append(piv_r.round(2).to_string())
    L.append(""); L.append("knife5:"); L.append((piv_k * 100).round(0).to_string())
    L.append("")

    L += ["## E. Small model — can knife5 be predicted? (logistic, train-fit, test-eval)", ""]
    feats = ["price_vs_ema200_pct", "rev_days_since_low_20", "rev_higher_low_pct",
             "rev_close_vs_ema20_pct", "rev_last_5d_return_pct", "rev_rsi14",
             "rev_decline_in_atr", "rev_down_up_vol_ratio_12", "rev_ema20_slope_5d_pct",
             "rev_dist_to_swing_low_20_pct"]
    dd = det.dropna(subset=feats + ["knife5", R]).copy()
    tr, te = dd[dd.split == "train"], dd[dd.split == "test"]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        sc = StandardScaler().fit(tr[feats])
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(tr[feats]), tr.knife5)
        p_tr = lr.predict_proba(sc.transform(tr[feats]))[:, 1]
        p_te = lr.predict_proba(sc.transform(te[feats]))[:, 1]
        L.append(f"- knife5 base rate: train {tr.knife5.mean():.1%}  test {te.knife5.mean():.1%}")
        L.append(f"- logistic AUC: train {roc_auc_score(tr.knife5, p_tr):.3f}  test {roc_auc_score(te.knife5, p_te):.3f}")
        for name, coef in sorted(zip(feats, lr.coef_[0]), key=lambda x: -abs(x[1])):
            L.append(f"    {name:<32}{coef:+.3f}")
        # decile of predicted knife-risk on test
        te = te.assign(pk=p_te, dec=pd.qcut(p_te, 5, labels=False, duplicates="drop"))
        L.append("")
        L.append("  test rows by predicted-knife-risk quintile (0=safest):")
        for q, g in te.groupby("dec"):
            L.append(f"    Q{q}: {m(g)}")
        # also predict R directly
        from sklearn.linear_model import LinearRegression
        lrr = LinearRegression().fit(sc.transform(tr[feats]), tr[R].clip(-1, 15))
        te = te.assign(pr=lrr.predict(sc.transform(te[feats])), rdec=lambda x: pd.qcut(x.pr, 5, labels=False, duplicates="drop"))
        L.append("")
        L.append("  test rows by predicted-R quintile (4=best):")
        for q, g in te.groupby("rdec"):
            L.append(f"    Q{q}: {m(g)}")
    except Exception as e:
        L.append(f"(sklearn unavailable or failed: {e})")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[supp] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
