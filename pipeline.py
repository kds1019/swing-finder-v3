"""
SwingFinder Agents — orchestrator and CLI entrypoint.

    Market Data Agent (Alpaca)   --+
    Research Agent (FMP)          -+--> Decision Agent (Claude) --> Ranked trade plans
    Portfolio Agent (Webull)      -+

Usage:
    python pipeline.py                          # full run (universe size is dynamic, built live from FMP)
    python pipeline.py --limit 20 --skip-decision   # fast smoke test, no research/Anthropic calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd

from config.settings import load_settings
from core.universe import build_universe
from core.sector_cap import apply_sector_cap, apply_sector_cap_to_picks
from core.pick_tracking import (
    load_pick_outcomes_log, save_pick_outcomes_log, score_due_picks,
    record_picks, compute_pick_accuracy_summary,
)
from agents.market_data_agent import MarketDataAgent, compute_market_bias
from agents.research_agent import ResearchAgent
from agents.portfolio_agent import PortfolioAgent
from agents.decision_agent import DecisionAgent, FINAL_WATCHLIST_SIZE
from core.price_reaction import annotate_news_with_reaction, recent_daily_bars

# Max tickers carried into the research/decision step, after the technical screener and the
# loose pre-research sector cap (settings.pre_research_sector_cap). DecisionAgent RANKS all of
# these on stabilisation + fundamentals/news; the real 3/sector diversification cap
# (settings.sector_cap) is then applied to that ranking and the top FINAL_WATCHLIST_SIZE
# survive. When more candidates pass the screener than this cap, the deepest pullbacks are
# kept (agents.market_data_agent sorts ranked_df by PriceVsEMA200Pct).
# Lowered 40 -> 30: the agent now writes a full research_highlight/rationale/bear_case for
# EVERY candidate (not just a 20-name pick list), ~3.7k output tokens each — 40 would push a
# full run past claude-sonnet-5's 128k output ceiling and truncate the JSON. 30 leaves
# headroom, and the calibration found no benefit to a wider pool anyway (see docs/strategy.md).
CANDIDATE_POOL_SIZE = 30

PICK_OUTCOMES_LOG_PATH = "pick_outcomes.csv"    # persisted in the repo, like results/

# core.trend_context fields to join onto each final pick, keyed by ticker. Attached here
# (in Python, post-hoc) rather than added to DecisionAgent's own JSON contract in
# agents/decision_agent.py — setup_type is deterministic, not an LLM judgment (unlike
# support_status), and this keeps the Decision Agent's prompt/output schema untouched.
# Purely additive to results/latest.json: new keys on each pick, nothing renamed/removed.
# Phase 3 (docs/strategy.md): setup_type now also feeds the Decision Agent's ranking
# (agents/decision_agent.py's SYSTEM_PROMPT) and apply_trend_context_trade_management() below
# — NOT the screener gate itself (core.pullback_reversal.detect_pullback_reversal is
# unchanged; a ticker that doesn't get trend_continuation or reversion_bounce still passes
# through exactly as before, just with setup_type=null and default trade management).
TREND_CONTEXT_PICK_FIELDS = {
    "TrendState": "trend_state",
    "SetupType": "setup_type",
    "RetracementPct": "retracement_pct",
    "InFibZone": "in_fib_zone",
    "TrendEMA50": "trend_ema50",
    "TrendEMA200": "trend_ema200",
    "PriceAboveTrendEMA50": "price_above_trend_ema50",
    "PriceAboveTrendEMA200": "price_above_trend_ema200",
    "TrendEMA200SlopePct": "trend_ema200_slope_pct",
    "TrendEMA200LongSlopePct": "trend_ema200_long_slope_pct",
    "SwingHigh": "swing_high",
    "SwingLow": "swing_low",
    "PullbackWidthBars": "pullback_width_bars",
    "RSI14": "rsi14",
    "RelVolume": "rel_volume",
}


def _na_to_none(v):
    """pandas' .to_dict() can hand back a bare NaN for a missing value in an otherwise
    string/object column (confirmed live: a mixed string/None SetupType column round-tripped
    through set_index(...).to_dict() came back as float('nan') for the None rows, not None
    itself) — json.dumps() then emits the literal (invalid-JSON) token NaN instead of null.
    pd.isna() also safely handles None/NaT, so this is used as a blanket sanitizer rather than
    a float-only isinstance check."""
    return None if pd.isna(v) else v


def attach_trend_context(picks: list[dict], features_df: pd.DataFrame) -> None:
    """Joins TREND_CONTEXT_PICK_FIELDS onto each pick dict in `picks`, in place, by ticker.
    No-op (leaves picks unchanged) if a ticker isn't found or a field wasn't computed for it
    (e.g. insufficient history for the 200-EMA) — never raises on missing trend context."""
    if not picks or features_df.empty or "Ticker" not in features_df.columns:
        return
    cols = [c for c in TREND_CONTEXT_PICK_FIELDS if c in features_df.columns]
    lookup = features_df.set_index("Ticker")[cols].to_dict(orient="index")
    for p in picks:
        row = lookup.get(p.get("ticker"), {})
        for src_col, dest_key in TREND_CONTEXT_PICK_FIELDS.items():
            p[dest_key] = _na_to_none(row.get(src_col))


def attach_short_interest(picks: list[dict], features_df: pd.DataFrame) -> None:
    """Joins agents.research_agent.ResearchAgent.get_short_interest()'s per-ticker dict onto
    each pick dict in `picks`, in place, by ticker — guaranteed visible in results/*.json
    regardless of whether the Decision Agent's prose happens to mention it (it also sees the
    raw ShortInterest field and reasons about it for ranking/flags — see decision_agent.py —
    but this is the deterministic passthrough of the actual numbers, same pattern as
    attach_trend_context above). No-op if a ticker isn't found or ShortInterest wasn't
    computed for it (e.g. the Nasdaq lookup failed) — never raises."""
    if not picks or features_df.empty or "Ticker" not in features_df.columns or "ShortInterest" not in features_df.columns:
        return
    lookup = dict(zip(features_df["Ticker"], features_df["ShortInterest"]))
    for p in picks:
        si = lookup.get(p.get("ticker")) or {}
        p["short_interest_shares"] = _na_to_none(si.get("short_interest_shares"))
        p["short_interest_change_pct"] = _na_to_none(si.get("short_interest_change_pct"))
        p["days_to_cover"] = _na_to_none(si.get("days_to_cover"))
        p["short_percent_of_float"] = _na_to_none(si.get("short_percent_of_float"))
        p["short_interest_settlement_date"] = _na_to_none(si.get("settlement_date"))


def attach_insider_activity(picks: list[dict], features_df: pd.DataFrame) -> None:
    """Joins agents.research_agent.ResearchAgent.summarize_insider_activity()'s per-ticker
    dict onto each pick dict in `picks`, in place, by ticker — same deterministic-passthrough
    pattern as attach_short_interest above. No-op if a ticker isn't found or InsiderActivity
    wasn't computed for it — never raises."""
    if not picks or features_df.empty or "Ticker" not in features_df.columns or "InsiderActivity" not in features_df.columns:
        return
    lookup = dict(zip(features_df["Ticker"], features_df["InsiderActivity"]))
    for p in picks:
        ia = lookup.get(p.get("ticker")) or {}
        p["insider_purchase_count"] = _na_to_none(ia.get("purchase_count"))
        p["insider_sale_count"] = _na_to_none(ia.get("sale_count"))
        p["insider_net_value"] = _na_to_none(ia.get("net_value"))
        p["insider_most_recent_purchase_date"] = _na_to_none(ia.get("most_recent_purchase_date"))
        p["insider_most_recent_sale_date"] = _na_to_none(ia.get("most_recent_sale_date"))


def apply_trend_context_trade_management(picks: list[dict]) -> None:
    """Phase 3 (docs/strategy.md): setup_type-aware exit mode, applied in place AFTER
    attach_trend_context() has already put `setup_type` on each pick.
      - exit_mode: "fixed_target" for reversion_bounce (trailing disabled downstream in
        core.pick_tracking.score_due_picks — a quick in-and-out, matching
        research/trend_context_backtest.py's bucketed exit); "trailing" for everything else
        (unchanged live default: +2R activate / trail peak-1R, never loosens). Target is
        still a ceiling in both modes; only whether the stop trails differs.
    Position sizing is deliberately NOT done here or anywhere in the pipeline (removed
    2026-09-30 per user instruction): the user sizes each trade at entry from their open
    positions, risk tolerance, and trade type. Never raises."""
    for p in picks or []:
        p["exit_mode"] = "fixed_target" if p.get("setup_type") == "reversion_bounce" else "trailing"

# ~1 quarter of calendar-day news — enough to judge the latest earnings reaction and any
# recent catalyst/trend, without the ~2yr blob the old 270 (+ a stale *2.5 buffer in
# fetch_news) produced, which was ~$1 of Decision Agent input tokens per run on its own.
NEWS_LOOKBACK_DAYS = 90


def apply_earnings_buffer(enriched_df: pd.DataFrame, settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hard-excludes only tickers reporting earnings the same day or next day
    (`earnings_buffer_exclude_days`, 1) — no stop can protect against an overnight
    gap through a print that close, so no qualitative case is strong enough to
    override it.

    Tickers reporting further out, up to `earnings_buffer_hard_days` (14), used to
    be dropped here unconditionally too; they're now kept and tagged with
    EarningsProximityTier ("earnings_imminent" within `earnings_buffer_soft_days`
    (7), else "earnings_upcoming") so the Decision Agent — which already has each
    ticker's EarningsHistory/IncomeGrowth/AnalystRating by this point — can make a
    real, deliberate call: a genuinely strong earnings setup (real beat streak,
    accelerating growth, bullish analyst consensus) can be picked as an explicit
    earnings-catalyst play instead of every near-term-earnings ticker being
    mechanically dropped regardless of how it actually looks.
    """
    if enriched_df.empty or "DaysToEarnings" not in enriched_df.columns:
        return enriched_df, enriched_df.iloc[0:0].copy()

    def is_hard_exclude(days) -> bool:
        return not pd.isna(days) and 0 <= days <= settings.earnings_buffer_exclude_days

    def proximity_tier(days) -> Optional[str]:
        # DaysToEarnings' map(earnings) can carry a ticker with no upcoming earnings date
        # as NaN (pandas coerces a dict of ints + None to float64 with NaN standing in for
        # None), not None itself — pd.isna() catches both, a plain `is None` check wouldn't.
        if pd.isna(days) or days > settings.earnings_buffer_hard_days:
            return None
        if days <= settings.earnings_buffer_soft_days:
            return "earnings_imminent"
        return "earnings_upcoming"

    hard_mask = enriched_df["DaysToEarnings"].apply(is_hard_exclude)
    excluded = enriched_df[hard_mask].copy()
    excluded["ExclusionReason"] = "earnings_same_day_or_next_day"
    kept = enriched_df[~hard_mask].copy()
    kept["EarningsProximityTier"] = kept["DaysToEarnings"].apply(proximity_tier)

    return kept, excluded


# A catalyst the Decision Agent labels "recent" must be at most this many calendar days old
# (by the catalyst_date it reports). Matches the prompt's own ~7-day definition, which it
# was observed ignoring (SRRK 2026-09-30: a 19-day-old FDA approval labelled "recent").
CATALYST_RECENT_MAX_DAYS = 7


# Flags the Decision Agent may add itself (judgment calls). Every other flag is computed in
# Python — compute_precomputed_flags() before the model runs (it sees them as input), plus
# StillFalling / CatalystStale after — so the flag vocabulary is closed and always correct.
# Before 2026-09-30 the model applied ~8 threshold flags itself; an audit of the 9/30 run
# found them correct but the rules cost ~15% of the prompt, and it invented ad-hoc flags
# (e.g. "AboveVolumePOC") outside any definition.
MODEL_JUDGMENT_FLAGS = ("CatalystFaded", "EarningsCatalyst")


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def compute_precomputed_flags(row, swing_sectors: set, open_order_symbols: set) -> list[str]:
    """Deterministic, threshold-based flags for one enriched shortlist row. Thresholds are
    the ones the Decision Agent prompt used to apply itself (unchanged), plus
    AtAnalystTarget (split out of TargetsBeingCut), EarningsSoon, SectorOverlap, OpenOrder."""
    flags: list[str] = []

    def _dict(v):
        return v if isinstance(v, dict) else {}   # a missing cell can arrive as NaN (truthy)

    si = _dict(row.get("ShortInterest"))
    spf, dtc, chg = (_num(si.get("short_percent_of_float")), _num(si.get("days_to_cover")),
                     _num(si.get("short_interest_change_pct")))
    if (spf is not None and spf >= 10) or (dtc is not None and dtc >= 5):
        flags.append("HeavilyShorted")
    if chg is not None and chg >= 10:
        flags.append("ShortsAdding")

    ia = _dict(row.get("InsiderActivity"))
    buys, sells = int(_num(ia.get("purchase_count")) or 0), int(_num(ia.get("sale_count")) or 0)
    if buys >= 1:
        flags.append("InsiderBuying")
    if sells >= 1 and buys == 0:
        flags.append("InsiderSelling")

    ar = _dict(row.get("AnalystRating"))
    rev, n = _num(ar.get("targetRevisionRecentPct")), int(_num(ar.get("lastMonthTargetCount")) or 0)
    if rev is not None and rev <= -8 and n >= 2:
        flags.append("TargetsBeingCut")
    tgt, px = _num(ar.get("lastMonthAvgTarget")), _num(row.get("Price"))
    if tgt and px and n >= 1 and px >= tgt:
        flags.append("AtAnalystTarget")

    poc_pct = _num(row.get("PriceVsPOCPct"))
    if poc_pct is not None and poc_pct > 0:
        flags.append("AboveVolumePOC")
    if row.get("WeakRR") in (True, 1) and not isinstance(row.get("WeakRR"), float):
        flags.append("WeakRR")
    if row.get("StopSanityFlag") in (True, 1) and not isinstance(row.get("StopSanityFlag"), float):
        flags.append("StopSanity")
    if isinstance(row.get("EarningsProximityTier"), str):
        flags.append("EarningsSoon")
    if row.get("Sector") in swing_sectors:
        flags.append("SectorOverlap")
    if row.get("Ticker") in open_order_symbols:
        flags.append("OpenOrder")
    return flags


def attach_precomputed_flags(final_df: pd.DataFrame, portfolio_context: dict,
                             sector_lookup: dict, excluded_tickers) -> pd.DataFrame:
    """Adds a PrecomputedFlags column (see compute_precomputed_flags). SectorOverlap counts
    only current SWING positions — the user's long-term holds (settings.excluded_tickers)
    are not swing exposure."""
    if final_df.empty:
        return final_df
    held = [pos.get("symbol") for pos in portfolio_context.get("positions") or []]
    swing_sectors = {sector_lookup.get(sym) for sym in held
                     if sym and sym not in set(excluded_tickers or ())} - {None}
    open_syms = {o.get("symbol") for o in portfolio_context.get("open_orders") or []} - {None}
    out = final_df.copy()
    out["PrecomputedFlags"] = [compute_precomputed_flags(r, swing_sectors, open_syms)
                               for r in out.to_dict(orient="records")]
    return out


def finalize_pick_fields(picks: list[dict], final_df: pd.DataFrame) -> None:
    """In place, right after the Decision Agent returns: entry/stop/target/rr_ratio come
    straight from the trade plan (the model no longer re-types numbers), and flags become
    PrecomputedFlags + StillFalling (from the model's support_status) + only the
    MODEL_JUDGMENT_FLAGS the model set. Any other model flag is dropped (logged)."""
    if not picks or final_df.empty:
        return
    rows = {r["Ticker"]: r for r in final_df.to_dict(orient="records")}
    for p in picks:
        r = rows.get(p.get("ticker"))
        model_flags = list(p.get("flags") or [])
        flags = list(r.get("PrecomputedFlags") or []) if r else []
        if r:
            p["entry"] = round(float(r["Price"]), 2) if _num(r.get("Price")) is not None else None
            p["stop"], p["target"], p["rr_ratio"] = r.get("Stop"), r.get("Target"), r.get("RRRatio")
        if p.get("support_status") == "still_falling":
            flags.append("StillFalling")
        flags += [f for f in model_flags if f in MODEL_JUDGMENT_FLAGS and f not in flags]
        dropped = [f for f in model_flags if f not in MODEL_JUDGMENT_FLAGS and f not in flags]
        if dropped:
            print(f"[pipeline] {p.get('ticker')}: dropped non-judgment model flags {dropped}",
                  file=sys.stderr)
        p["flags"] = flags


def attach_price_context(enriched_df: pd.DataFrame, bars_by_ticker: dict) -> pd.DataFrame:
    """Adds RecentDailyBars and annotates every News item with its real price reaction
    (core.price_reaction) — so the Decision Agent reads what the tape did after a catalyst
    instead of inferring it from a headline. Pure passthrough of already-fetched bars; no
    extra API calls. Tickers without bars get [] / None fields rather than an error."""
    if enriched_df.empty:
        return enriched_df
    out = enriched_df.copy()
    out["RecentDailyBars"] = out["Ticker"].map(lambda t: recent_daily_bars(bars_by_ticker.get(t)))
    if "News" in out.columns:
        out["News"] = [
            annotate_news_with_reaction(news or [], bars_by_ticker.get(t))
            for t, news in zip(out["Ticker"], out["News"])
        ]
    return out


def enforce_catalyst_recency(picks: list[dict], now: Optional[pd.Timestamp] = None) -> None:
    """Deterministic check on the Decision Agent's catalyst_status, in place: a "recent"
    label needs a catalyst_date within CATALYST_RECENT_MAX_DAYS. Otherwise it's downgraded
    to "none" with a "CatalystStale" flag, and the model's original label is kept in
    catalyst_status_model for auditing. Never raises."""
    now = (now or pd.Timestamp.now()).normalize()
    for p in picks:
        if p.get("catalyst_status") != "recent":
            continue
        d = pd.to_datetime(p.get("catalyst_date"), errors="coerce")
        if pd.notna(d) and (now - d.normalize()).days <= CATALYST_RECENT_MAX_DAYS:
            continue
        p["catalyst_status_model"] = "recent"
        p["catalyst_status"] = "none"
        flags = list(p.get("flags") or [])
        if "CatalystStale" not in flags:
            flags.append("CatalystStale")
        p["flags"] = flags


def run_pipeline(
    limit: int | None = None,
    random_sample: bool = False,
    skip_decision: bool = False,
    dry_run: bool = True,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    decision_input_out: Optional[str] = None,
) -> dict:
    """decision_input_out: if set, write the exact Decision Agent user payload to this path
    and return BEFORE calling the model (no picks logged) — used by
    .github/workflows/prompt_ab.yml to compare system prompts on identical input."""
    settings = load_settings()

    universe = build_universe(settings)
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

    # --- Loose pre-research sector cap (cost control only) ---
    # NOT the real diversification limit — that is settings.sector_cap, applied to the
    # Decision Agent's ranked output below. This only stops one selling-off sector from
    # consuming the whole candidate pool / FMP budget in a broad sector pullback.
    pooled_df, pre_research_excluded_df = apply_sector_cap(ranked_df, settings.pre_research_sector_cap)
    print(f"[pipeline] After pre-research sector cap ({settings.pre_research_sector_cap}/sector): "
          f"{len(pooled_df)} tickers ({len(pre_research_excluded_df)} held back)", file=sys.stderr)

    # Candidate pool for the research/decision step — DecisionAgent ranks all of these; the
    # 3/sector diversification cap is applied afterward (see CANDIDATE_POOL_SIZE).
    shortlist_df = pooled_df.head(candidate_pool_size).reset_index(drop=True)

    if skip_decision:
        return {
            "shortlist": json.loads(shortlist_df.to_json(orient="records")),
            "pre_research_sector_excluded": json.loads(pre_research_excluded_df.to_json(orient="records")),
            "market_bias": market_bias,
            "skipped_decision": True,
        }

    # --- Research Agent: VIX gate + shortlist enrichment (fundamentals, analyst ratings,
    # earnings-beat/miss history, quarterly growth trend, ~1 quarter of news, and a derived
    # catalyst-recency signal) ---
    research_agent = ResearchAgent(settings)
    vix = research_agent.get_vix_level()
    market_gate_open = vix is not None and vix <= settings.vix_gate_ceiling
    print(f"[pipeline] VIX={vix} gate_ceiling={settings.vix_gate_ceiling} gate_open={market_gate_open}", file=sys.stderr)

    enriched_df = research_agent.enrich_shortlist(
        shortlist_df, market_agent=market_agent, news_lookback_days=NEWS_LOOKBACK_DAYS
    )
    enriched_df = attach_price_context(enriched_df, bars_by_ticker)
    final_df, earnings_excluded_df = apply_earnings_buffer(enriched_df, settings)
    print(f"[pipeline] After earnings buffer: {len(final_df)} tickers "
          f"({len(earnings_excluded_df)} excluded)", file=sys.stderr)

    # --- Portfolio Agent: existing positions/sector-exposure/open-orders context ---
    portfolio_agent = PortfolioAgent(settings)
    positions_df = portfolio_agent.get_positions()
    open_orders_df = portfolio_agent.get_open_orders()
    open_orders = portfolio_agent.flatten_open_orders(open_orders_df)
    sector_lookup = dict(zip(universe["Ticker"], universe["Sector"]))
    sector_exposure = portfolio_agent.check_sector_exposure(positions_df, sector_lookup)

    portfolio_context = {
        "positions": json.loads(positions_df.to_json(orient="records")) if not positions_df.empty else [],
        "sector_exposure": sector_exposure,
        "open_orders": open_orders,
    }
    final_df = attach_precomputed_flags(final_df, portfolio_context, sector_lookup, settings.excluded_tickers)

    # --- Pick outcome tracking (part 1): score past picks before this run's synthesis, so
    # the Decision Agent can see its own historical win rate before making new calls. ---
    pick_log = load_pick_outcomes_log(PICK_OUTCOMES_LOG_PATH)
    pick_log = score_due_picks(pick_log, market_agent)
    pick_track_record = compute_pick_accuracy_summary(pick_log)
    print(f"[pipeline] Pick track record: {pick_track_record}", file=sys.stderr)

    if decision_input_out:
        payload = DecisionAgent._build_user_prompt(
            final_df, portfolio_context, market_gate_open, pick_track_record,
        )
        with open(decision_input_out, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"[pipeline] Saved Decision Agent input ({len(final_df)} candidates) to "
              f"{decision_input_out}; stopping before the model call.", file=sys.stderr)
        return {"decision_input_saved": decision_input_out, "candidates": len(final_df)}

    # --- Decision Agent: research-driven RANKING of every candidate ---
    decision_agent = DecisionAgent(settings)
    result = decision_agent.synthesize(
        final_df, portfolio_context, market_gate_open, pick_track_record,
    )

    # --- Diversification cap: keep the 3 highest-RANKED names per sector, then take the top
    # FINAL_WATCHLIST_SIZE. This runs here, on the Decision Agent's quality ranking, not on the
    # raw screener output — so the survivors are a sector's best candidates, not the first the
    # technical screener happened to surface. ---
    sector_capped_out: list[dict] = []
    if isinstance(result, dict) and result.get("ranked_picks"):
        finalize_pick_fields(result["ranked_picks"], final_df)
        enforce_catalyst_recency(result["ranked_picks"])
        sector_lookup = dict(zip(final_df["Ticker"], final_df["Sector"]))
        kept, sector_capped_out = apply_sector_cap_to_picks(
            result["ranked_picks"], sector_lookup, settings.sector_cap
        )
        result["ranked_picks"] = kept[:FINAL_WATCHLIST_SIZE]
        result["sector_capped_out"] = sector_capped_out
        print(f"[pipeline] Sector cap ({settings.sector_cap}/sector) on Decision Agent ranking: "
              f"{len(result['ranked_picks'])} final picks ({len(sector_capped_out)} capped out)",
              file=sys.stderr)

        # Trend-context fields (setup_type/trend_state/etc.), then Phase 3's setup_type-aware
        # trade management (exit_mode + reversion_bounce sizing) — see TREND_CONTEXT_PICK_FIELDS
        # and apply_trend_context_trade_management above. Trade management only applies to the
        # actual final picks, not sector_capped_out (those aren't being recommended for entry).
        attach_trend_context(result["ranked_picks"], final_df)
        attach_trend_context(result["sector_capped_out"], final_df)
        attach_short_interest(result["ranked_picks"], final_df)
        attach_short_interest(result["sector_capped_out"], final_df)
        attach_insider_activity(result["ranked_picks"], final_df)
        attach_insider_activity(result["sector_capped_out"], final_df)
        apply_trend_context_trade_management(result["ranked_picks"])

    # --- Pick outcome tracking (part 2): log this run's final (post-cap) picks for scoring. ---
    ranked_picks = result.get("ranked_picks", []) if isinstance(result, dict) else []
    # final_df still carries core.pullback_reversal's per-ticker measurements — pass it so
    # each logged pick records how it matched the screener (see docs/strategy.md calibration).
    pick_log = record_picks(
        pick_log, ranked_picks, pd.Timestamp.now().strftime("%Y-%m-%d"), features_df=final_df
    )
    save_pick_outcomes_log(pick_log, PICK_OUTCOMES_LOG_PATH)

    return {
        "market_bias": market_bias,
        "vix": vix,
        "market_gate_open": market_gate_open,
        "pre_research_sector_excluded_count": len(pre_research_excluded_df),
        "sector_capped_out_count": len(sector_capped_out),
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
    parser.add_argument("--save-decision-input", type=str, default=None, metavar="PATH",
                        help="Write the Decision Agent's exact input payload to PATH and stop before calling it")
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
            decision_input_out=args.save_decision_input,
        )
    finally:
        sys.stdout.flush()
        os.dup2(real_stdout_fd, 1)
        os.close(real_stdout_fd)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
