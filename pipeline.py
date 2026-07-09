"""
SwingFinder Agents — orchestrator and CLI entrypoint.

    Market Data Agent (Alpaca)   --+
    Research Agent (FMP)          -+--> Decision Agent (Claude) --> Ranked trade plans
    Portfolio Agent (Webull)      -+

Usage:
    python pipeline.py                          # full 945-ticker run
    python pipeline.py --limit 20 --skip-decision   # fast smoke test, no FMP/Anthropic calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from config.settings import load_settings
from core.universe import load_universe
from core.sector_cap import apply_sector_cap
from core.ml_forecast import ensemble_ml_forecast, evaluate_ml_edge_score
from core.multi_timeframe import get_multi_timeframe_analysis
from core.relative_strength import calculate_relative_strength_rank
from core.volume_profile import evaluate_volume_profile_position
from core.patterns import detect_patterns, evaluate_pattern_score
from core.ml_tracking import (
    load_predictions_log, save_predictions_log, score_due_predictions,
    record_predictions, compute_accuracy_summary,
)
from agents.market_data_agent import MarketDataAgent, compute_market_bias
from agents.research_agent import ResearchAgent
from agents.portfolio_agent import PortfolioAgent
from agents.decision_agent import DecisionAgent

SHORTLIST_SIZE = 20  # max tickers carried past sector cap into Decision Agent synthesis

ML_PREDICTIONS_LOG_PATH = "ml_predictions.csv"  # persisted in the repo, like results/


DEEP_HISTORY_LOOKBACK_DAYS = 750  # ~3yrs — the universe scan's 60-day bars are too shallow for
                                   # ml_forecast (wants up to 1500 bars) / weekly MTF resampling
                                   # (wants ~730 days). Fine to re-fetch deeper history here since
                                   # this only runs on the ~30-ticker shortlist, not the full universe.

VOLUME_PROFILE_WINDOW_DAYS = 60  # trailing window within the deep bars used for the POC histogram


def enrich_with_technical_analysis(
    shortlist_df: pd.DataFrame,
    market_agent: MarketDataAgent,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Optional enrichment (core.ml_forecast / core.multi_timeframe / core.relative_strength /
    core.volume_profile / core.patterns) on the post-sector-cap shortlist only — never the
    full universe, since the ML ensemble trains a fresh Random Forest + Gradient Boosting per
    ticker and needs much deeper history than the universe-scan bars provide. Volume profile
    and pattern detection are themselves cheap, but are computed here too (on this same
    deep-history fetch) rather than in the full universe scan, so there's one consistent
    reading per ticker instead of two different ones from two different data windows.

    Re-scores SmartScore with volume-profile-position, ML-edge, and chart-pattern adjustments
    (bonus/penalty, same pattern as core.deep_discount_filter) and re-sorts by the result —
    this only affects ranking among tickers that already survived sector cap on their
    pre-enrichment SmartScore; it can't influence sector cap membership, since these
    adjustments aren't available until after this deep-history fetch.
    """
    if shortlist_df.empty:
        return shortlist_df

    from core.indicators import compute_indicators

    tickers = shortlist_df["Ticker"].tolist()
    deep_bars = market_agent.fetch_universe_bars(tickers + ["SPY"], lookback_days=DEEP_HISTORY_LOOKBACK_DAYS)
    spy_deep_bars = deep_bars.get("SPY")

    df = shortlist_df.copy()
    ml_forecasts, mtf_analyses, rs_ranks = [], [], []
    vp_adjustments, ml_adjustments, pattern_adjustments = [], [], []

    for ticker, price in zip(df["Ticker"], df["Price"]):
        bars = deep_bars.get(ticker)

        if bars is not None:
            ml_result = ensemble_ml_forecast(compute_indicators(bars.copy()), vix_df=vix_df)
            ml_forecasts.append(ml_result)
            mtf_analyses.append(get_multi_timeframe_analysis(bars))
            rs_ranks.append(
                calculate_relative_strength_rank(ticker, bars, spy_deep_bars, period=60)
                if spy_deep_bars is not None else None
            )
            vp_adjustments.append(evaluate_volume_profile_position(bars, window=VOLUME_PROFILE_WINDOW_DAYS))
            ml_adjustments.append(evaluate_ml_edge_score(ml_result, float(price)))
            pattern_adjustments.append(evaluate_pattern_score(detect_patterns(bars)))
        else:
            ml_forecasts.append({"success": False, "error": "no bars"})
            mtf_analyses.append(None)
            rs_ranks.append(None)
            vp_adjustments.append({"triggered": False, "score_adjustment": 0, "flag": None, "poc": None, "price_vs_poc_pct": None})
            ml_adjustments.append({"triggered": False, "score_adjustment": 0, "flag": "ml_edge_unavailable", "edge_pct": None})
            pattern_adjustments.append({"triggered": False, "score_adjustment": 0, "flag": None,
                                         "pattern_name": None, "pattern_confidence": None, "pattern_action": None})

    df["MLForecast"] = ml_forecasts
    df["MultiTimeframe"] = mtf_analyses
    df["RelativeStrength"] = rs_ranks
    df["VolumeProfilePOC"] = [a["poc"] for a in vp_adjustments]
    df["PriceVsPOCPct"] = [a["price_vs_poc_pct"] for a in vp_adjustments]
    df["VolumeProfileFlag"] = [a["flag"] for a in vp_adjustments]
    df["MLEdgePct"] = [a["edge_pct"] for a in ml_adjustments]
    df["MLEdgeFlag"] = [a["flag"] for a in ml_adjustments]
    df["PatternName"] = [a["pattern_name"] for a in pattern_adjustments]
    df["PatternConfidence"] = [a["pattern_confidence"] for a in pattern_adjustments]
    df["PatternAction"] = [a["pattern_action"] for a in pattern_adjustments]
    df["PatternFlag"] = [a["flag"] for a in pattern_adjustments]

    df["SmartScore"] = [
        max(0, min(100, score + vp["score_adjustment"] + ml["score_adjustment"] + pat["score_adjustment"]))
        for score, vp, ml, pat in zip(df["SmartScore"], vp_adjustments, ml_adjustments, pattern_adjustments)
    ]
    df = df.sort_values("SmartScore", ascending=False).reset_index(drop=True)

    return df


def apply_earnings_buffer(enriched_df: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard-excludes tickers with earnings within `earnings_buffer_hard_days` (14).
    Within the excluded set, tags whether earnings are "imminent" (within
    `earnings_buffer_soft_days`, 7) vs merely "upcoming" (8-14 days) — this
    gives both configured thresholds real meaning without contradiction
    (a single "days_to_earnings <= N" drop can't use two different N's on the
    same ticker at once, so the softer threshold becomes a severity tier on
    the harder one's exclusions instead).
    """
    if enriched_df.empty or "DaysToEarnings" not in enriched_df.columns:
        return enriched_df, enriched_df.iloc[0:0].copy()

    def is_hard_exclude(days) -> bool:
        return days is not None and 0 <= days <= settings.earnings_buffer_hard_days

    def severity(days) -> str:
        if days is not None and days <= settings.earnings_buffer_soft_days:
            return "earnings_imminent"
        return "earnings_upcoming"

    hard_mask = enriched_df["DaysToEarnings"].apply(is_hard_exclude)
    excluded = enriched_df[hard_mask].copy()
    excluded["ExclusionReason"] = excluded["DaysToEarnings"].apply(severity)
    kept = enriched_df[~hard_mask].copy()

    return kept, excluded


def run_pipeline(
    limit: int | None = None,
    skip_decision: bool = False,
    skip_ml: bool = False,
    dry_run: bool = True,
    shortlist_size: int = SHORTLIST_SIZE,
) -> dict:
    settings = load_settings()

    universe = load_universe(settings.universe_csv_path)
    if limit:
        universe = universe.head(limit)

    print(f"[pipeline] Universe loaded: {len(universe)} tickers", file=sys.stderr)

    # --- Market Data Agent: full-universe SmartScore scan ---
    market_agent = MarketDataAgent(settings)
    spy_bars = market_agent.fetch_spy_bars(settings.bars_lookback_days)
    market_bias = compute_market_bias(spy_bars)
    print(f"[pipeline] Market bias (SPY EMA20 vs EMA50): {market_bias}", file=sys.stderr)

    ranked_df, bars_by_ticker = market_agent.scan_universe(universe, settings, market_bias)
    print(f"[pipeline] SmartScore'd {len(ranked_df)} / {len(universe)} tickers with a signal", file=sys.stderr)

    if ranked_df.empty:
        return {"error": "No tickers produced a SmartScore signal", "ranked_df_empty": True}

    # --- Sector cap ---
    capped_df, sector_excluded_df = apply_sector_cap(ranked_df, settings.sector_cap)
    print(f"[pipeline] After sector cap ({settings.sector_cap}/sector): {len(capped_df)} tickers "
          f"({len(sector_excluded_df)} excluded)", file=sys.stderr)

    shortlist_df = capped_df.head(shortlist_size).reset_index(drop=True)

    # --- Optional technical enrichment: ML ensemble forecast, multi-timeframe alignment,
    # relative strength vs SPY. Runs on the shortlist only. Skippable for fast iteration
    # since the ML ensemble trains a fresh model per ticker. ---
    vix_df = None
    ml_track_record = None
    if not skip_ml:
        if settings.fmp_api_key:
            try:
                vix_history_agent = ResearchAgent(settings)
                end = pd.Timestamp.now()
                start = end - pd.Timedelta(days=DEEP_HISTORY_LOOKBACK_DAYS + 5)
                vix_df = vix_history_agent.get_vix_history(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            except Exception as e:
                # VIX is an optional ml_forecast feature (see core/ml_forecast.py) — any
                # failure here (bad/placeholder key, network issue) should degrade to
                # "no VIX feature" rather than abort the whole enrichment step.
                print(f"[pipeline] VIX history fetch failed, proceeding without it: {e}", file=sys.stderr)
        shortlist_df = enrich_with_technical_analysis(shortlist_df, market_agent, vix_df=vix_df)
        print(f"[pipeline] Technical enrichment (ML/MTF/RS/Patterns) added for {len(shortlist_df)} tickers", file=sys.stderr)

        # --- ML forecast accuracy tracking: score any past predictions whose 5-trading-day
        # window has now elapsed, then log this run's new forecasts for future scoring. ---
        ml_log = load_predictions_log(ML_PREDICTIONS_LOG_PATH)
        ml_log = score_due_predictions(ml_log, market_agent)
        new_predictions = [
            {
                "prediction_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "ticker": row["Ticker"],
                "entry_price": float(row["Price"]),
                "predicted_return_pct": row["MLEdgePct"],
                "predicted_price": row["MLForecast"]["ensemble_price"],
                "confidence": row["MLForecast"]["confidence"],
                "days_ahead": 5,
            }
            for _, row in shortlist_df.iterrows()
            if isinstance(row["MLForecast"], dict) and row["MLForecast"].get("success")
        ]
        ml_log = record_predictions(ml_log, new_predictions)
        save_predictions_log(ml_log, ML_PREDICTIONS_LOG_PATH)
        ml_track_record = compute_accuracy_summary(ml_log)
        print(f"[pipeline] ML track record: {ml_track_record}", file=sys.stderr)

    if skip_decision:
        return {
            "shortlist": json.loads(shortlist_df.to_json(orient="records")),
            "sector_excluded": json.loads(sector_excluded_df.to_json(orient="records")),
            "market_bias": market_bias,
            "ml_track_record": ml_track_record,
            "skipped_decision": True,
        }

    # --- Research Agent: VIX gate + shortlist enrichment ---
    research_agent = ResearchAgent(settings)
    vix = research_agent.get_vix_level()
    market_gate_open = vix is not None and vix <= settings.vix_gate_ceiling
    print(f"[pipeline] VIX={vix} gate_ceiling={settings.vix_gate_ceiling} gate_open={market_gate_open}", file=sys.stderr)

    enriched_df = research_agent.enrich_shortlist(shortlist_df)
    final_df, earnings_excluded_df = apply_earnings_buffer(enriched_df, settings)
    print(f"[pipeline] After earnings buffer: {len(final_df)} tickers "
          f"({len(earnings_excluded_df)} excluded)", file=sys.stderr)

    # --- Portfolio Agent: existing positions/balance context ---
    portfolio_agent = PortfolioAgent(settings)
    positions_df = portfolio_agent.get_positions()
    balance = portfolio_agent.get_account_balance()
    sector_lookup = dict(zip(universe["Ticker"], universe["Sector"]))
    sector_exposure = portfolio_agent.check_sector_exposure(positions_df, sector_lookup)

    portfolio_context = {
        "positions": json.loads(positions_df.to_json(orient="records")) if not positions_df.empty else [],
        "balance": balance,
        "sector_exposure": sector_exposure,
    }

    # --- Decision Agent: final synthesis ---
    decision_agent = DecisionAgent(settings)
    result = decision_agent.synthesize(final_df, final_df, portfolio_context, market_gate_open, ml_track_record)

    return {
        "market_bias": market_bias,
        "vix": vix,
        "market_gate_open": market_gate_open,
        "sector_excluded_count": len(sector_excluded_df),
        "earnings_excluded_count": len(earnings_excluded_df),
        "decision": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SwingFinder Agents pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit universe to first N tickers (fast iteration)")
    parser.add_argument("--skip-decision", action="store_true", help="Stop before the Anthropic call (test agents 1-3 only)")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML forecast/multi-timeframe/relative-strength enrichment (faster iteration)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Portfolio execution dry-run (default: True)")
    parser.add_argument("--shortlist-size", type=int, default=SHORTLIST_SIZE, help="Max tickers to carry past sector cap")
    args = parser.parse_args()

    # webull-openapi-python-sdk writes its auth/token diagnostic logs directly to a file
    # descriptor bound to the real stdout — confirmed this bypasses Python-level logging
    # reconfiguration (setLevel/removeHandler on the "webull" logger had no effect, so its
    # handler must be holding its own reference to the underlying fd rather than going through
    # the standard logging hierarchy). That silently corrupted
    # `python pipeline.py | tee results/latest.json` in GitHub Actions: those log lines landed
    # ahead of the final JSON in the committed results file, breaking anything trying to
    # json.load() it (including this project's own GitHub-connector-based result reads).
    # OS-level fd redirection is the only thing that reliably stops it regardless of how any
    # dependency internally opens/binds its log stream — swap fd 1 to point at fd 2 for the
    # run, then restore it before printing the actual result.
    real_stdout_fd = os.dup(1)
    sys.stdout.flush()
    os.dup2(2, 1)
    try:
        result = run_pipeline(
            limit=args.limit,
            skip_decision=args.skip_decision,
            skip_ml=args.skip_ml,
            dry_run=args.dry_run,
            shortlist_size=args.shortlist_size,
        )
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout_fd, 1)
        os.close(real_stdout_fd)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
