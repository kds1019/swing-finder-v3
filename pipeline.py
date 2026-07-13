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
from core.pick_tracking import (
    load_pick_outcomes_log, save_pick_outcomes_log, score_due_picks,
    record_picks, compute_pick_accuracy_summary,
)
from agents.market_data_agent import MarketDataAgent, compute_market_bias
from agents.research_agent import ResearchAgent
from agents.portfolio_agent import PortfolioAgent
from agents.decision_agent import DecisionAgent

# Max tickers (after the technical screener + sector cap) carried into the research/decision
# step — wider than the old SHORTLIST_SIZE=20, since DecisionAgent's job is now to SELECT the
# final ~18 (see agents.decision_agent.FINAL_WATCHLIST_SIZE) from this candidate pool using
# fundamentals/news, not just polish an already-fixed list. Ordered by BounceOffLowPct
# (core.pullback_reversal's own sort) if more candidates pass than this cap.
CANDIDATE_POOL_SIZE = 40

PICK_OUTCOMES_LOG_PATH = "pick_outcomes.csv"    # persisted in the repo, like results/

NEWS_LOOKBACK_DAYS = 270  # ~9 months — within the user's requested 6-12 month research window


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
    random_sample: bool = False,
    skip_decision: bool = False,
    dry_run: bool = True,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
) -> dict:
    settings = load_settings()

    universe = load_universe(settings.universe_csv_path)
    if limit:
        # random_sample=True picks limit tickers at random instead of the first limit rows —
        # .head(limit) is whatever order the universe CSV happens to be in (e.g. alphabetical),
        # not representative; useful for a fast deterministic smoke test, but a poor sample for
        # actually exercising the pullback/reversal screener against a real cross-section of the
        # universe. No fixed seed — each run gets a fresh random sample, unlike research/'s
        # reproducibility-focused sampling (research/walk_forward_backtest.py's own
        # select_sample_universe), since there's no cross-run comparison need here.
        universe = universe.sample(n=min(limit, len(universe))) if random_sample else universe.head(limit)

    print(f"[pipeline] Universe loaded: {len(universe)} tickers"
          f"{' (random sample)' if (limit and random_sample) else ''}", file=sys.stderr)

    # --- Market Data Agent: full-universe technical screen (core.pullback_reversal) ---
    market_agent = MarketDataAgent(settings)
    spy_bars = market_agent.fetch_spy_bars(settings.bars_lookback_days)
    market_bias = compute_market_bias(spy_bars)
    print(f"[pipeline] Market bias (SPY EMA20 vs EMA50): {market_bias}", file=sys.stderr)

    ranked_df, bars_by_ticker = market_agent.scan_universe(universe, settings)
    print(f"[pipeline] Pullback/reversal screener matched {len(ranked_df)} / {len(universe)} tickers", file=sys.stderr)

    if ranked_df.empty:
        return {"error": "No tickers matched the pullback/reversal screener", "ranked_df_empty": True}

    # Drop the user's existing long-term holds — never swing candidates, and excluded here
    # (before sector cap/research) so they don't consume a sector-cap slot or an FMP call.
    if settings.excluded_tickers:
        excluded_mask = ranked_df["Ticker"].isin(settings.excluded_tickers)
        if excluded_mask.any():
            print(f"[pipeline] Excluding long-term holds from screener matches: "
                  f"{ranked_df.loc[excluded_mask, 'Ticker'].tolist()}", file=sys.stderr)
        ranked_df = ranked_df[~excluded_mask].reset_index(drop=True)

    if ranked_df.empty:
        return {"error": "No tickers matched the pullback/reversal screener after excluding long-term holds", "ranked_df_empty": True}

    # --- Sector cap ---
    capped_df, sector_excluded_df = apply_sector_cap(ranked_df, settings.sector_cap)
    print(f"[pipeline] After sector cap ({settings.sector_cap}/sector): {len(capped_df)} tickers "
          f"({len(sector_excluded_df)} excluded)", file=sys.stderr)

    # Candidate pool for the research/decision step — DecisionAgent selects the final
    # watchlist from this, it isn't already a fixed-size shortlist (see CANDIDATE_POOL_SIZE).
    shortlist_df = capped_df.head(candidate_pool_size).reset_index(drop=True)

    if skip_decision:
        return {
            "shortlist": json.loads(shortlist_df.to_json(orient="records")),
            "sector_excluded": json.loads(sector_excluded_df.to_json(orient="records")),
            "market_bias": market_bias,
            "skipped_decision": True,
        }

    # --- Research Agent: VIX gate + shortlist enrichment (fundamentals, analyst ratings,
    # earnings-beat/miss history, quarterly growth trend, and 6-12mo news) ---
    research_agent = ResearchAgent(settings)
    vix = research_agent.get_vix_level()
    market_gate_open = vix is not None and vix <= settings.vix_gate_ceiling
    print(f"[pipeline] VIX={vix} gate_ceiling={settings.vix_gate_ceiling} gate_open={market_gate_open}", file=sys.stderr)

    enriched_df = research_agent.enrich_shortlist(
        shortlist_df, market_agent=market_agent, news_lookback_days=NEWS_LOOKBACK_DAYS
    )
    final_df, earnings_excluded_df = apply_earnings_buffer(enriched_df, settings)
    print(f"[pipeline] After earnings buffer: {len(final_df)} tickers "
          f"({len(earnings_excluded_df)} excluded)", file=sys.stderr)

    # --- Portfolio Agent: existing positions/balance/open-orders context ---
    portfolio_agent = PortfolioAgent(settings)
    positions_df = portfolio_agent.get_positions()
    balance = portfolio_agent.get_account_balance()
    open_orders_df = portfolio_agent.get_open_orders()
    open_orders = portfolio_agent.flatten_open_orders(open_orders_df)
    sector_lookup = dict(zip(universe["Ticker"], universe["Sector"]))
    sector_exposure = portfolio_agent.check_sector_exposure(positions_df, sector_lookup)

    portfolio_context = {
        "positions": json.loads(positions_df.to_json(orient="records")) if not positions_df.empty else [],
        "balance": balance,
        "sector_exposure": sector_exposure,
        "open_orders": open_orders,
    }

    # --- Pick outcome tracking (part 1): score past picks before this run's synthesis, so
    # the Decision Agent can see its own historical win rate before making new calls. ---
    pick_log = load_pick_outcomes_log(PICK_OUTCOMES_LOG_PATH)
    pick_log = score_due_picks(pick_log, market_agent)
    pick_track_record = compute_pick_accuracy_summary(pick_log)
    print(f"[pipeline] Pick track record: {pick_track_record}", file=sys.stderr)

    # --- Decision Agent: research-driven selection of the final watchlist ---
    decision_agent = DecisionAgent(settings)
    result = decision_agent.synthesize(
        final_df, portfolio_context, market_gate_open, pick_track_record, settings.risk_per_trade_pct,
    )

    # --- Pick outcome tracking (part 2): log this run's new picks for future scoring. ---
    ranked_picks = result.get("ranked_picks", []) if isinstance(result, dict) else []
    pick_log = record_picks(pick_log, ranked_picks, pd.Timestamp.now().strftime("%Y-%m-%d"))
    save_pick_outcomes_log(pick_log, PICK_OUTCOMES_LOG_PATH)

    return {
        "market_bias": market_bias,
        "vix": vix,
        "market_gate_open": market_gate_open,
        "sector_excluded_count": len(sector_excluded_df),
        "earnings_excluded_count": len(earnings_excluded_df),
        "decision": result,
        "pick_track_record": pick_track_record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SwingFinder Agents pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit universe to N tickers (fast iteration)")
    parser.add_argument("--random-sample", action="store_true",
                         help="With --limit, pick N tickers at random instead of the first N in the universe CSV")
    parser.add_argument("--skip-decision", action="store_true", help="Stop before FMP research/Anthropic calls (test screener + sector cap only)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Portfolio execution dry-run (default: True)")
    parser.add_argument("--candidate-pool-size", type=int, default=CANDIDATE_POOL_SIZE,
                         help="Max tickers (after screener + sector cap) carried into research/decision")
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
            random_sample=args.random_sample,
            skip_decision=args.skip_decision,
            dry_run=args.dry_run,
            candidate_pool_size=args.candidate_pool_size,
        )
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout_fd, 1)
        os.close(real_stdout_fd)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
