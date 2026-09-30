"""Tests for the Python side of the Decision Agent contract (pipeline.py):
compute_precomputed_flags / attach_precomputed_flags (threshold flags computed before the
model runs) and finalize_pick_fields (trade-plan numbers + closed flag vocabulary after it).
Values are real ones from the 2026-09-30 scan where noted.

Run: python -m tests.test_decision_contract   (or pytest, if installed)
"""

import pandas as pd

from pipeline import (annotate_summary_names, attach_precomputed_flags, compute_precomputed_flags,
                      compute_support_check, finalize_pick_fields)


def _row(**kw):
    base = {"Ticker": "X", "Sector": "Technology", "Price": 50.0, "Stop": 48.0, "Target": 60.0,
            "RRRatio": 5.0, "WeakRR": False, "StopSanityFlag": False, "PriceVsPOCPct": -3.0,
            "EarningsProximityTier": None, "ShortInterest": {}, "InsiderActivity": {},
            "AnalystRating": {}}
    base.update(kw)
    return base


def test_short_interest_thresholds_match_real_930_cases():
    # CGNX: 4.49% float but 5.01 days to cover -> HeavilyShorted; -7.3% change -> not adding
    cgnx = _row(ShortInterest={"short_percent_of_float": 4.49, "days_to_cover": 5.01,
                               "short_interest_change_pct": -7.3})
    assert compute_precomputed_flags(cgnx, set(), set()) == ["HeavilyShorted"]
    # ALGT: 9.85% float, 4.82 DTC -> not heavily shorted; +16.9% -> ShortsAdding
    algt = _row(ShortInterest={"short_percent_of_float": 9.85, "days_to_cover": 4.82,
                               "short_interest_change_pct": 16.9})
    assert compute_precomputed_flags(algt, set(), set()) == ["ShortsAdding"]
    # no data (NYSE-listed) -> no short flags, never "no shorts"
    assert compute_precomputed_flags(_row(), set(), set()) == []


def test_insider_analyst_and_misc_flags():
    r = _row(InsiderActivity={"purchase_count": 0, "sale_count": 4},
             AnalystRating={"targetRevisionRecentPct": -13.6, "lastMonthTargetCount": 3,
                            "lastMonthAvgTarget": 49.0},
             PriceVsPOCPct=1.2, WeakRR=True, EarningsProximityTier="earnings_upcoming",
             Sector="Industrials", Ticker="DAL2")
    got = compute_precomputed_flags(r, {"Industrials"}, {"DAL2"})
    assert got == ["InsiderSelling", "TargetsBeingCut", "AtAnalystTarget", "AboveVolumePOC",
                   "WeakRR", "EarningsSoon", "SectorOverlap", "OpenOrder"], got
    # one analyst cutting is too weak a signal for TargetsBeingCut
    weak = _row(AnalystRating={"targetRevisionRecentPct": -20, "lastMonthTargetCount": 1})
    assert "TargetsBeingCut" not in compute_precomputed_flags(weak, set(), set())
    buy = _row(InsiderActivity={"purchase_count": 2, "sale_count": 6})
    assert compute_precomputed_flags(buy, set(), set()) == ["InsiderBuying"]


def test_sector_overlap_ignores_long_term_holds():
    df = pd.DataFrame([_row(Ticker="ABC", Sector="Healthcare"), _row(Ticker="DEF", Sector="Technology")])
    ctx = {"positions": [{"symbol": "HELP"}, {"symbol": "MCHP"}], "open_orders": [{"symbol": "DEF"}]}
    out = attach_precomputed_flags(df, ctx, {"HELP": "Healthcare", "MCHP": "Technology"}, ("HELP",))
    assert out.loc[0, "PrecomputedFlags"] == []                            # HELP is a long-term hold
    assert out.loc[1, "PrecomputedFlags"] == ["SectorOverlap", "OpenOrder"]


def test_finalize_pick_fields():
    df = pd.DataFrame([_row(Ticker="AAA", Price=60.3312, Stop=57.71, Target=73.43, RRRatio=5.0,
                            PrecomputedFlags=["HeavilyShorted"]),
                       _row(Ticker="BBB", PrecomputedFlags=[])])
    picks = [{"ticker": "AAA", "support_status": "confirmed", "entry": 1.0,
              "flags": ["CatalystFaded", "AboveVolumePOC", "HeavilyShorted"]},
             {"ticker": "BBB", "support_status": "still_falling", "flags": None}]
    finalize_pick_fields(picks, df)
    a, b = picks
    assert (a["entry"], a["stop"], a["target"], a["rr_ratio"]) == (60.33, 57.71, 73.43, 5.0)
    assert a["flags"] == ["HeavilyShorted", "CatalystFaded"]   # invented AboveVolumePOC dropped
    assert b["flags"] == ["StillFalling"]


def test_nan_cells_do_not_crash_or_false_flag():
    import numpy as np
    r = _row(ShortInterest=float("nan"), InsiderActivity=np.nan, AnalystRating=None,
             EarningsProximityTier=np.nan, WeakRR=np.True_)
    assert compute_precomputed_flags(r, set(), set()) == ["WeakRR"]


def _stab(t, rcr, duv, r5, r10, tier="stabilising"):
    return {"Ticker": t, "KnifeRiskTier": tier, "RangeContractionRatio": rcr,
            "DownUpVolumeRatio": duv, "Last5dReturnPct": r5, "Last10dReturnPct": r10}


def test_support_check_matches_user_rule_on_real_930_values():
    # one non-severe failure -> keep confirmed (the CSCO/CGNX case the user chose option 1 for)
    assert compute_support_check(_stab("CSCO", 0.73, 1.45, 1.25, 0.05))["suggested_support"] == "confirmed"
    assert compute_support_check(_stab("CGNX", 0.66, 1.62, 4.31, 0.90))["suggested_support"] == "confirmed"
    assert compute_support_check(_stab("WTTR", 1.30, 0.99, 2.28, -2.76))["suggested_support"] == "confirmed"
    # severe range alone -> downgrade
    assert compute_support_check(_stab("PNW", 1.65, 0.81, 1.58, -1.55))["suggested_support"] == "forming"
    # still sliding on both horizons (severe trend) -> downgrade
    algt = compute_support_check(_stab("ALGT", 0.96, 0.99, -2.69, -4.54))
    assert algt["trend"] == "severe" and algt["suggested_support"] == "forming"
    # two failures -> downgrade; forming tier drops to still_falling
    assert compute_support_check(_stab("GEO", 1.25, 1.66, -0.46, -3.02))["suggested_support"] == "forming"
    assert compute_support_check(_stab("X", 1.2, 1.4, 1.0, 1.0, tier="forming"))["suggested_support"] == "still_falling"
    # missing data never fails a check
    assert compute_support_check(_stab("N", None, None, None, None))["suggested_support"] == "confirmed"


def test_summary_names_not_in_final_list_get_a_note():
    r = {"overall_recommendation": "Favor CSCO and FLEX; avoid WERN.",
         "ranked_picks": [{"ticker": "FLEX"}]}
    annotate_summary_names(r, ["CSCO", "FLEX", "WERN", "ON"])
    assert r["overall_recommendation"].endswith(
        "[Pipeline note: CSCO, WERN are named above but not in the final list "
        "(removed by the sector cap or excluded).]"), r
    clean = {"overall_recommendation": "Gate open; ONE theme dominates.", "ranked_picks": []}
    annotate_summary_names(clean, ["ON"])           # word boundary: "ONE" is not "ON"
    assert "Pipeline note" not in clean["overall_recommendation"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
