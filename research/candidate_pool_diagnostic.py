"""
RESEARCH / AUDIT ONLY — does not touch the live pipeline.

Diagnoses two design questions about the funnel that happens BEFORE the Decision Agent ever
sees a candidate, raised directly (2026-09-13) after a live run came back with zero
trend_continuation picks among the 30 tickers actually reviewed:

  1. Is SetupType-blind ordering actually losing trend_continuation candidates before
     research? Today's screener orders candidates by KnifeRiskTier then pullback depth
     (agents/market_data_agent.py's scan_universe) — SetupType plays no role in who survives
     the pre-research sector cap (8/sector, cost control only) or the CANDIDATE_POOL_SIZE
     (30) cut that follow. A trend_continuation candidate sitting behind a deeper/more-
     stabilised reversion_bounce candidate in the same sector or pool slot gets silently
     dropped before the Decision Agent — which DOES know to rank trend_continuation higher —
     ever gets a vote.
  2. Does the pre-research sector cap actually protect the FINAL result, or is it dead
     weight? It exists so one over-represented sector can't consume the whole 30-slot
     research budget in a way that would ALSO starve the final 3/sector diversification cap
     of candidates from other sectors. This checks whether that's a real, common risk or a
     rare edge case, and whether removing it costs anything in practice.

Uses today's live-gate signal set (research/data/current_trend_signals.pkl) as the historical
stand-in for what agents/market_data_agent.py's scan_universe would have produced each day —
tags every signal with TrendState/InFibZone/SetupType (computed the same way
core.trend_context does it, reusing the identical functions) since the cached signal set
doesn't carry those fields. Zero FMP/Anthropic calls — this is a pure funnel/composition
diagnostic on deterministic price-derived fields, not a P&L backtest, and it does NOT
simulate the Decision Agent's actual re-ranking (no live LLM call, and the point-in-time-data
problem discussed for backtesting the Decision Agent applies here too) — the "final list"
figures below use the SAME funnel ordering as a structural proxy for what the Decision Agent
would rank, which measures the MECHANICAL effect of the pre-research design, not the true
final-output quality after real research/ranking.

Usage:  python -m research.candidate_pool_diagnostic
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.indicators import compute_indicators
from core.sector_cap import apply_sector_cap
from core.trend_context import classify_setup_type, compute_trend_state, measure_swing_fib_retracement
from research.current_trend_gate_ab import SIG_CACHE as CURRENT_TREND_SIG_CACHE

BARS_DIR = Path(__file__).resolve().parent / "data" / "bars"
SIG_CACHE = Path(__file__).resolve().parent / "data" / "candidate_pool_signals.pkl"
OUT = Path(__file__).resolve().parent / "candidate_pool_diagnostic.md"

PRE_RESEARCH_SECTOR_CAP = 8   # settings.pre_research_sector_cap default
CANDIDATE_POOL_SIZE = 30      # pipeline.py's CANDIDATE_POOL_SIZE default
FINAL_SECTOR_CAP = 3          # settings.sector_cap default
FINAL_WATCHLIST_SIZE = 20     # agents/decision_agent.py's FINAL_WATCHLIST_SIZE

TIER_RANK = {"stabilising": 0, "forming": 1, "still_falling": 3}
SETUP_RANK = {"trend_continuation": 0, "reversion_bounce": 1}


def tag_signals(sig: pd.DataFrame) -> pd.DataFrame:
    """Adds TrendState/InFibZone/SetupType to each cached signal, computed the same way
    core.trend_context does it live — as of that signal's own date, using only bars up to
    and including it (no lookahead)."""
    out_rows = []
    for t, g in sig.groupby("ticker"):
        raw = pd.read_pickle(BARS_DIR / f"{t}.pkl")
        df_full = compute_indicators(raw.copy()).set_index("Date")
        for _, row in g.iterrows():
            prefix = df_full.loc[:row["date"]]
            trend = compute_trend_state(prefix)
            fib = measure_swing_fib_retracement(prefix)
            setup_type = classify_setup_type(
                trend.get("trend_state"), fib.get("in_fib_zone"), row["tier"] == "stabilising",
            )
            out_rows.append({**row.to_dict(), "trend_state": trend.get("trend_state"),
                              "in_fib_zone": fib.get("in_fib_zone"), "setup_type": setup_type})
    return pd.DataFrame(out_rows)


def simulate_day(day_df: pd.DataFrame, sort_cols: list[str], use_pre_research_cap: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replays the real funnel (core.sector_cap.apply_sector_cap, same function pipeline.py
    uses) for one day's candidate set under one ordering + one pre-research-cap setting.
    Returns (research_pool, final_list)."""
    ordered = day_df.sort_values(sort_cols)
    if use_pre_research_cap:
        pooled, _ = apply_sector_cap(ordered, PRE_RESEARCH_SECTOR_CAP, sector_col="sector")
    else:
        pooled = ordered
    research_pool = pooled.head(CANDIDATE_POOL_SIZE)
    final_list, _ = apply_sector_cap(research_pool, FINAL_SECTOR_CAP, sector_col="sector")
    final_list = final_list.head(FINAL_WATCHLIST_SIZE)
    return research_pool, final_list


def main():
    if not CURRENT_TREND_SIG_CACHE.exists():
        sys.exit("research/data/current_trend_signals.pkl not found — run "
                  "`python -m research.current_trend_gate_ab` first.")

    if SIG_CACHE.exists():
        sig = pd.read_pickle(SIG_CACHE)
        print(f"[pool_diag] loaded {len(sig)} cached tagged signals", file=sys.stderr)
    else:
        base = pd.read_pickle(CURRENT_TREND_SIG_CACHE)
        base["date"] = pd.to_datetime(base["date"]).dt.normalize()
        print(f"[pool_diag] tagging {len(base)} signals with TrendState/SetupType...", file=sys.stderr)
        sig = tag_signals(base)
        SIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
        sig.to_pickle(SIG_CACHE)
        print(f"[pool_diag] tagged + cached {len(sig)} signals", file=sys.stderr)

    print(f"[pool_diag] setup_type counts: {sig['setup_type'].value_counts(dropna=False).to_dict()}",
          file=sys.stderr)

    sig["_tier_rank"] = sig["tier"].map(TIER_RANK).fillna(2)
    sig["_setup_rank"] = sig["setup_type"].map(SETUP_RANK).fillna(2)

    ORDERINGS = {
        "current (tier, depth)": ["_tier_rank", "depth"],
        "setup-priority (setup, tier, depth)": ["_setup_rank", "_tier_rank", "depth"],
    }

    day_groups = list(sig.groupby("date"))
    n_days = len(day_groups)
    print(f"[pool_diag] {n_days} distinct screener-match days", file=sys.stderr)

    # Per-day raw diagnostics (ordering-independent)
    raw_rows = []
    for day, g in day_groups:
        sector_counts = g["sector"].value_counts()
        raw_rows.append({
            "date": day, "n_matches": len(g),
            "has_tc": bool((g["setup_type"] == "trend_continuation").any()),
            "max_sector_share": sector_counts.iloc[0] / len(g) if len(g) else 0.0,
            "n_sectors": g["sector"].nunique(),
        })
    raw_df = pd.DataFrame(raw_rows)

    # Per-day, per-(ordering, pre-cap on/off) funnel simulation
    results = {}
    for ord_name, sort_cols in ORDERINGS.items():
        for use_cap, cap_name in [(True, "with pre-research cap"), (False, "no pre-research cap")]:
            rows = []
            for day, g in day_groups:
                research_pool, final_list = simulate_day(g, sort_cols, use_cap)
                rows.append({
                    "date": day,
                    "tc_reaches_research": bool((research_pool["setup_type"] == "trend_continuation").any()),
                    "tc_reaches_final": bool((final_list["setup_type"] == "trend_continuation").any()),
                    "research_pool_size": len(research_pool),
                    "research_pool_n_sectors": research_pool["sector"].nunique(),
                    "final_list_size": len(final_list),
                })
            results[(ord_name, cap_name)] = pd.DataFrame(rows).set_index("date")

    days_with_tc = raw_df[raw_df["has_tc"]]["date"]
    print(f"[pool_diag] {len(days_with_tc)}/{n_days} days had >=1 raw trend_continuation match",
          file=sys.stderr)

    L = ["# Candidate-pool funnel diagnostic — SetupType-blind ordering + pre-research sector "
         "cap (RESEARCH)", "",
         "Pure price-data diagnostic (zero FMP/Anthropic calls) replaying the funnel that runs "
         "BEFORE the Decision Agent ever sees a candidate: KnifeRiskTier-then-depth ordering, "
         "an 8/sector pre-research cap, then a 30-candidate pool size cut. Uses today's "
         "live-gate signal set as the historical stand-in for each day's raw screener matches, "
         "tagged retroactively with TrendState/SetupType. 'final list' below uses the SAME "
         "funnel ordering as a structural proxy for the Decision Agent's ranking (no real LLM "
         "call, no point-in-time research data) — it measures the MECHANICAL effect of this "
         "design, not real final-output quality.",
         f"- {n_days} distinct screener-match days; {len(days_with_tc)} of them "
         f"({len(days_with_tc)/n_days*100:.0f}%) had at least one raw trend_continuation match",
         f"- setup_type counts across all signals: {sig['setup_type'].value_counts(dropna=False).to_dict()}",
         "",
         "## Question 1 — does SetupType-blind ordering actually lose trend_continuation "
         "candidates before research?", "",
         "On days with >=1 raw trend_continuation match, % of those days it survived to reach "
         "the research pool / the (proxy) final list, by ordering (holding the pre-research "
         "cap ON, today's actual setting):", "",
         "| ordering | trades reaches research | reaches final list |",
         "|---|---|---|"]
    for ord_name in ORDERINGS:
        r = results[(ord_name, "with pre-research cap")].loc[days_with_tc]
        L.append(f"| {ord_name} | {r['tc_reaches_research'].mean()*100:.0f}% | "
                 f"{r['tc_reaches_final'].mean()*100:.0f}% |")
    L += ["", "## Question 2 — does the pre-research sector cap matter, and does removing it "
          "cost anything?", "",
          "### Raw sector concentration (before any cap — how skewed are the day's matches?)",
          "", "| stat | value |", "|---|---|",
          f"| mean max-sector-share | {raw_df['max_sector_share'].mean()*100:.0f}% |",
          f"| median max-sector-share | {raw_df['max_sector_share'].median()*100:.0f}% |",
          f"| p90 max-sector-share | {raw_df['max_sector_share'].quantile(0.9)*100:.0f}% |",
          f"| days with max-sector-share > 50% | "
          f"{(raw_df['max_sector_share'] > 0.5).mean()*100:.0f}% |", "",
          "### Effect of removing the pre-research cap (current ordering held fixed)", "",
          "| | research pool size | research pool # sectors | final list size |",
          "|---|---|---|---|"]
    for cap_name in ["with pre-research cap", "no pre-research cap"]:
        r = results[("current (tier, depth)", cap_name)]
        L.append(f"| {cap_name} | {r['research_pool_size'].mean():.1f} avg | "
                 f"{r['research_pool_n_sectors'].mean():.1f} avg | "
                 f"{r['final_list_size'].mean():.1f} avg |")
    L.append("")

    # Same breakdown restricted to the most sector-concentrated days, where the cap should
    # matter most if it matters at all.
    concentrated_days = raw_df[raw_df["max_sector_share"] > raw_df["max_sector_share"].quantile(0.75)]["date"]
    L += [f"### Same comparison, restricted to the most sector-concentrated quartile of days "
          f"({len(concentrated_days)} days, max-sector-share > "
          f"{raw_df['max_sector_share'].quantile(0.75)*100:.0f}%)", "",
          "| | research pool size | research pool # sectors | final list size |",
          "|---|---|---|---|"]
    for cap_name in ["with pre-research cap", "no pre-research cap"]:
        r = results[("current (tier, depth)", cap_name)].loc[concentrated_days]
        L.append(f"| {cap_name} | {r['research_pool_size'].mean():.1f} avg | "
                 f"{r['research_pool_n_sectors'].mean():.1f} avg | "
                 f"{r['final_list_size'].mean():.1f} avg |")
    L.append("")

    L += ["## All four combinations — trend_continuation survival to final list", "",
          "| ordering | pre-research cap | reaches research | reaches final list |",
          "|---|---|---|---|"]
    for ord_name in ORDERINGS:
        for cap_name in ["with pre-research cap", "no pre-research cap"]:
            r = results[(ord_name, cap_name)].loc[days_with_tc]
            L.append(f"| {ord_name} | {cap_name} | {r['tc_reaches_research'].mean()*100:.0f}% | "
                     f"{r['tc_reaches_final'].mean()*100:.0f}% |")
    L.append("")

    L += ["_Signal set is today's live gate only (G1-G4 + current-trend slope gate + weak-RR "
          "dropped) — matches what agents/market_data_agent.py would find, not the full raw "
          "universe before any live gate. Sector labels are today's live universe applied "
          "retroactively (same limitation as every other backtest in this repo)._"]

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[pool_diag] wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
