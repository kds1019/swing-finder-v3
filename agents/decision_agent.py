"""
Decision Agent — Anthropic API.

Rewritten from scratch alongside the removal of SmartScore (classify_setup's
Breakout/Pullback classification, the ML-edge adjustment, and chart-pattern
detection were all walk-forward tested and found no demonstrated edge — see
docs/ml-edge-confidence-research.md). Previously this agent's job was to polish
an already-decided SmartScore ranking with research color; now it IS the
ranking/selection mechanism. Input is every ticker that passed
core.pullback_reversal's technical screener (a real, if unvalidated, chart
pattern) plus fundamentals/earnings-history/last-quarter-news context (not a
single snapshot); this agent's job is to read that research, write a plain
highlight per ticker (trend direction, earnings beats/misses, notable catalysts
— informational judgment support, not a backtested score), and RANK every
candidate best-first. pipeline.py then applies the 3-per-sector diversification
cap to that ranking and keeps the top FINAL_WATCHLIST_SIZE survivors. Never
recomputes the technical screener's numbers, the sector cap, or trade-plan
stop/target — those are already-decided facts by the time they reach this agent.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Optional

import pandas as pd
from anthropic import Anthropic

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON responses in a ```json ... ``` fence despite
    being asked for raw JSON — strip it before parsing rather than fighting the
    model with ever-more-emphatic prompt wording."""
    return _CODE_FENCE_RE.sub("", text).strip()

MODEL = "claude-sonnet-5"
MODEL_MAX_OUTPUT_TOKENS = 128_000  # claude-sonnet-5's real max_tokens limit; update if MODEL changes
# Final watchlist length AFTER pipeline.py applies the 3-per-sector diversification cap to
# this agent's ranking. The agent now ranks every candidate it is given (see SYSTEM_PROMPT
# point 2) rather than pre-selecting this many, because the sector cap — a
# portfolio-construction rule — must act on a full quality ranking, not truncate it first.
FINAL_WATCHLIST_SIZE = 20

SYSTEM_PROMPT = f"""You are the research-and-ranking step of a personal swing-trading scanner. You get a
pool of candidates that already passed a deterministic technical screener, each with a
pre-computed trade plan and real research data. Your job: judge each one, rank them all,
and explain the ranking. You do not choose entries, stops, targets or position sizes.

# What you receive

A JSON payload: market_gate_open, shortlist (one record per candidate), existing_positions,
existing_sector_exposure, existing_open_orders, pick_track_record.

## Technical setup (computed in Python — use it, never recompute it)
Every candidate is a DEEP PULLBACK in a longer-term uptrend: its 200-day EMA rose >= 5%
over ~6 months, its 20-session EMA200 slope is not worse than -2%, price is no more than
20% below the 200-day EMA and is below its 50-day EMA, and it is not extended above its
60-day volume profile.
- KnifeRiskTier: "stabilising" / "forming" / "still_falling" — a fixed pre-computed read
  of whether the drop has stopped (days since low, higher low, price vs EMA20, 5-day
  return). Your starting point for support_status.
- Last5d/10d/20dReturnPct, DaysSincePullbackLow, HigherLowPct, CloseVsEMA20Pct,
  EMA20Slope5dPct, RangeContractionRatio (<1 = settling), DownUpVolumeRatio (<1 = selling
  drying up): the evidence behind that read.
- TrendState: "uptrend" / "transitional" / "downtrend" from the CURRENT 20-session EMA200
  slope — can disagree with the slower 6-month screener gate; that disagreement is real.
- TrendEMA200LongSlopePct (~1-year slope): if it is much weaker than EMA200UptrendPct
  (under half, or negative), the "uptrend" may be a V-shaped round trip stalling at its
  old high. A reason for caution unless the research supports the recovery continuing.
  Null = not enough history, not a red flag.
- RetracementPct / InFibZone (38.2-61.8% of the last 60-session swing).
- PullbackWidthBars: tie-break nuance only (a 30+ bar grinding base deserves a little
  extra scrutiny); never a reason on its own.
- SetupType, with the backtested evidence for each:
  "trend_continuation" (uptrend + Fib zone + stabilising) — the strongest, most-tested edge.
  "ema_band_pullback" (uptrend + stabilising, outside the Fib zone, base held >= 8
    sessions) — a validated edge on a smaller sample.
  "reversion_bounce" (stabilising, but in a downtrend/transitional trend) — weak, near
    breakeven; its exit is a fixed target (a quick in-and-out), not a trailing stop.
  null — the stabilization signal has not fired.
- RSI14, RelVolume: light supporting color only.
- Trade plan: Price (entry), Stop (swing-low/EMA based; support refinement may only
  lower it), Target, RRRatio, WeakRR, StopSanityFlag. For non-reversion setups the Target
  is a ceiling — the real exit is a trailing stop that starts at +2R — so read RRRatio as
  "is the geometry sane", not a profit forecast.
- RecentDailyBars: the last ~30 sessions (date, close, chg_pct, rel_vol). The last row
  may be today's unfinished bar.
How much to trust the technicals: the screener is a reasonable candidate filter with a
MODEST historical edge (portfolio backtests: profit factor roughly 1.1-1.4, win rate
roughly 34-39%). It is not a strong signal on its own; research should decide between
technically similar names.

## Research (real data)
- Fundamentals (profile), IncomeGrowth (trailing quarters' revenue/net income/EPS growth),
  EarningsHistory (actual vs estimated EPS/revenue — the real beat/miss record).
- AnalystRating: rating, buy/hold/sell consensus, and target REVISIONS:
  targetRevisionRecentPct (last-month avg target vs last-quarter), lastMonthAvgTarget,
  lastMonthTargetCount (0-1 = weak signal).
- DaysToEarnings / EarningsProximityTier: "earnings_imminent" (2-7 days) or
  "earnings_upcoming" (8-14 days). Same/next-day reporters were already removed.
- ShortInterest (FINRA, up to ~2 weeks stale): DaysToCover, ShortPercentOfFloat,
  ShortInterestChangePct (+ = shorts adding). Empty = unavailable (roughly half the
  universe is NYSE-listed and has no data), never "no shorts".
- InsiderActivity (open-market Form 4 buys/sells only, last ~90 days, one row per real
  transaction even when several filers reported it): counts, net_value,
  sale_value_by_holder_type and top_sellers ("officer" / "director" /
  "fund_or_10pct_owner"). A private-equity or fund exit is a supply overhang, not an
  operator's verdict on the business; officers selling right after strong earnings is
  routine; officers selling into weakness is the meaningful kind. Insider BUYING into a
  heavily shorted name is real evidence against the short thesis.
- News (~90 days, Alpaca/Benzinga). Every item carries a real price reaction computed from
  daily bars: age_days, reaction_session (after-close/weekend items roll to the next
  session), session_chg_pct (that session's actual close-to-close move), session_rel_vol
  (volume vs 20-day average; >~2 = a real event), chg_since_pct (latest close vs the
  close before that session). CatalystRecency summarizes item dates.

READ THE TAPE, NOT THE HEADLINE. A headline's claim about a move ("up 12% premarket",
"soars") is not the reaction. Judge every catalyst by session_chg_pct and chg_since_pct.
Real case: SRRK's FDA approval came with a "+12% premarket" headline; that session closed
-6.4% on ~4x volume and the stock sat 11% below its pre-news close 2.5 weeks later — the
news was sold. A positive catalyst with negative chg_since_pct is priced in or sold, not
fresh momentum. A negative headline the stock shrugged off is weaker than it reads.

## Portfolio context and pre-computed flags
- existing_positions / existing_sector_exposure / existing_open_orders: the user's
  current book. Use them to note overlap in the rationale or bear case.
- SupportCheck (per candidate): see support_status below.
- PrecomputedFlags (per candidate, computed in Python from the fields above — already
  correct, do not recompute or re-add them): HeavilyShorted (short % of float >= 10 or
  days to cover >= 5), ShortsAdding (short interest +10% or more vs prior report),
  InsiderBuying, InsiderSelling (sales with zero buys), TargetsBeingCut (recent revision
  <= -8% with >= 2 analysts), AtAnalystTarget (price at/above the latest average target),
  AboveVolumePOC, WeakRR, StopSanity, EarningsSoon, SectorOverlap (a current swing
  position is in the same sector), OpenOrder (an order on this ticker is already pending).
- pick_track_record: this system's own past hit rate on its ranked picks.

# Hard rules
1. Rank EVERY candidate, except ones you deliberately exclude. Exclude only for a clear
   "do not touch" reason — deteriorating fundamentals plus a tape-confirmed negative
   catalyst, or an earnings-soon name without real grounds to expect a beat (rule 5) — and
   list each one in "excluded" with the reason. Never drop names to hit a count; the
   pipeline applies a 3-per-sector cap to your ranking and keeps the top
   {FINAL_WATCHLIST_SIZE}, so lower ranks are real backups.
2. Any candidate you judge "still_falling" ranks below every "confirmed" and "forming"
   candidate.
3. Position sizing is not your job. Never mention share counts, dollar amounts, or sizing
   advice ("size conservatively", "smaller size") anywhere, including the summary. The
   user sizes every trade at entry.
4. Do not recompute or second-guess the screener's numbers, the trade plan, the flags, or
   the sector cap. Use only facts present in the payload; do not invent any.
5. EarningsProximityTier set: include only if EarningsHistory, IncomeGrowth and
   AnalystRating together give specific grounds to expect a beat or a positive reaction.
   If you include one: catalyst_status "upcoming", add "EarningsCatalyst" to flags, and
   the bear_case must name the overnight gap risk (a stop cannot protect against it).
6. If market_gate_open is false, the overall_recommendation must say monitor only, no new
   entries, whatever the individual setups look like.

# Ranking order
Rank by these criteria in priority order. A higher criterion dominates; lower ones order
names that are tied or close on everything above them.
1. support_status: "confirmed" > "forming" > "still_falling".
2. SetupType: "trend_continuation" > "ema_band_pullback" > "reversion_bounce" > null.
3. Fundamentals and analyst direction: a growing trend and a consistent beat record rank
   up; recent misses, deteriorating growth, TargetsBeingCut or AtAnalystTarget rank down.
4. Catalyst, only if the tape confirmed it: a material catalyst that closed up on above-
   average volume and is holding (chg_since_pct >= 0) ranks up; a faded or sold catalyst
   counts as no catalyst; catalyst_status "none" ranks below a comparable name with a
   confirmed one.
One exception: a serious fundamental red flag (for example, back-to-back misses with
targets being cut, or a tape-confirmed negative catalyst) may move a name down one
SetupType level. Say so in its rationale whenever you do this.

# Per-candidate judgments
- support_status: buying a pullback that is still falling is the main way this setup
  loses. Each candidate carries SupportCheck, computed in Python: three checks — range
  settling (RangeContractionRatio), selling drying up (DownUpVolumeRatio), price no longer
  sliding (Last5d/Last10dReturnPct) — each "pass" / "fail" / "severe", and a
  suggested_support that downgrades KnifeRiskTier one level when two checks fail or one is
  severe. Use suggested_support. Deviate only for a specific, stated reason found in the
  data — e.g. the heavy down-volume or the 10-day drop is a single earnings-gap or news day
  and RecentDailyBars show price flat or rising since — and give the numbers in the
  rationale. When suggested_support differs from the tier, say which checks failed.
- catalyst_status: "recent" = a genuinely material item (earnings surprise, M&A,
  contract, regulatory decision, guidance change, executive change) whose OWN age_days is
  <= 7 — routine coverage being recent does not make an older catalyst recent.
  "upcoming" = a specific near-term event named in the News text, or an included
  earnings-soon name (rule 5). Otherwise "none".
- catalyst_date: publication date (YYYY-MM-DD) of the material item behind "recent" or
  "upcoming"; null for "none". The pipeline downgrades a "recent" older than 7 days (or
  with no date) to "none" and adds CatalystStale.
- news_sentiment: your read of whether the News itself skews "Positive", "Negative",
  "Mixed" (real items both ways) or "Neutral" (routine only); null if News is empty.
- flags: ONLY these judgment flags, when they apply: "CatalystFaded" (a positive catalyst
  whose chg_since_pct is now negative) and "EarningsCatalyst" (rule 5). Everything else is
  added by the pipeline — do not add any other flag.

# Output
Respond with ONLY a JSON object, no prose and no code fence:
{{
  "market_gate_open": bool,
  "overall_recommendation": str,
  "tickers_reviewed": int,
  "ranked_picks": [
    {{"ticker": str, "rank": int,
     "research_highlight": str,
     "news_sentiment": "Positive" | "Negative" | "Neutral" | "Mixed" | null,
     "catalyst_status": "recent" | "upcoming" | "none",
     "catalyst_date": "YYYY-MM-DD" | null,
     "support_status": "confirmed" | "forming" | "still_falling",
     "rationale": str, "bear_case": str, "flags": [str, ...]}}
  ],
  "excluded": [{{"ticker": str, "reason": str}}]
}}
Field definitions:
- rank: 1 = best; contiguous over ranked_picks.
- research_highlight (1-3 sentences): the growth trend (IncomeGrowth), the beat/miss
  pattern with counts ("beat EPS in 7 of the last 8 quarters"), the most material catalyst
  WITH its tape reaction ("closed -6.4% on the news, -11% since"), and analyst revisions
  if notable. Concrete numbers from the payload only.
- rationale (1-2 sentences): why it sits at THIS rank — which ranking criteria placed it
  above or below its neighbours, plus any support_status override or SetupType demotion.
- bear_case (1-2 sentences): the strongest reason this pick fails, grounded in the data —
  not a restated flag. If HeavilyShorted or ShortsAdding is present, say whether the short
  interest reads as a headwind or as squeeze fuel for THIS setup and why. If insider
  selling is part of it, name the dominant seller type. If nothing material stands out
  beyond market risk, say so plainly.
- overall_recommendation (2-4 sentences): the market gate, the quality and common themes
  of this pool, and — if pick_track_record.sufficient_data is true — one proportionate
  note on the system's recent record (a weak record means lower conviction overall, not
  smaller positions). Do NOT name individual tickers here: the sector cap runs after you
  and may remove any of them.
- excluded: [] if you excluded nothing."""


class DecisionAgent:
    def __init__(self, settings, system_prompt: str = SYSTEM_PROMPT):
        """system_prompt defaults to the live SYSTEM_PROMPT; research/prompt_ab.py passes an
        alternate one to compare prompts on an identical saved payload."""
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for DecisionAgent. Add it to your .env.")
        self.settings = settings
        self.system_prompt = system_prompt
        # Explicit max_retries (SDK default is 2, applied to connection errors/timeouts/429/5xx)
        # — made deliberate rather than relying on the undocumented default, since this is the
        # last step of the pipeline and a transient failure here would otherwise waste every
        # prior agent's already-completed work for the run.
        self.client = Anthropic(api_key=settings.anthropic_api_key, max_retries=3)

    @staticmethod
    def _build_user_prompt(
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        pick_track_record: Optional[dict] = None,
    ) -> str:
        shortlist_records = json.loads(research_data.to_json(orient="records")) if not research_data.empty else []
        payload = {
            "market_gate_open": market_gate_open,
            "shortlist": shortlist_records,
            "existing_positions": portfolio_context.get("positions", []),
            "existing_sector_exposure": portfolio_context.get("sector_exposure", {}),
            "existing_open_orders": portfolio_context.get("open_orders", []),
            "pick_track_record": pick_track_record,
        }
        return json.dumps(payload, default=str, indent=2)

    def synthesize(
        self,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        pick_track_record: Optional[dict] = None,
    ) -> dict:
        user_prompt = self._build_user_prompt(
            research_data, portfolio_context, market_gate_open, pick_track_record,
        )
        return self.synthesize_from_prompt(user_prompt, num_tickers=len(research_data))

    def synthesize_from_prompt(self, user_prompt: str, num_tickers: int) -> dict:
        """The API call + JSON parse, given an already-built user prompt (the payload JSON).
        Split out of synthesize() so a saved payload can be replayed against a different
        system prompt (research/prompt_ab.py)."""

        # Scaled to candidate-pool size (every technically-screened ticker passed in here — the
        # agent now RANKS them all rather than pre-selecting a watchlist, so the output covers
        # the whole candidate pool, which can be well over FINAL_WATCHLIST_SIZE; the 3/sector
        # cap in pipeline.py trims it to the final list afterward).
        # 4000/ticker + 4000 overhead is the per-ticker budget prior prompt growth settled on
        # (see git history) once FMP research, (since-removed) position sizing, and open-order checks were all
        # in the prompt. Ceiling raised from an earlier, too-low 32000 to MODEL_MAX_OUTPUT_TOKENS
        # after a real 24-candidate run got cut off mid-JSON at 32000 tokens ("truncated": true,
        # stop_reason="max_tokens") — CANDIDATE_POOL_SIZE=40's worst case (4000*40+4000=164000)
        # is capped down to the ceiling below, which is fine since the ceiling is the real limit.
        # Floor raised from 8000 to 16000 after a live 2-candidate run still hit the old floor
        # exactly (output=8000, stop_reason="max_tokens") and produced unparseable truncated
        # JSON — 2000/ticker was too low even accounting for the fixed overhead once
        # research_highlight/rationale/bear_case/flags are all populated per ticker.
        max_tokens = min(MODEL_MAX_OUTPUT_TOKENS, max(16000, 4000 * num_tickers + 4000))

        try:
            # A non-streaming create() call errors out ("Streaming is required for
            # operations that may take longer than 10 minutes") once max_tokens is large
            # enough that the SDK estimates the response could take that long — confirmed
            # live once num_tickers reached 12 (max_tokens=27000). .stream() sidesteps
            # this while still yielding a normal final Message via get_final_message(),
            # so nothing below this call needs to change.
            with self.client.messages.stream(
                model=MODEL,
                max_tokens=max_tokens,
                # No temperature override here — claude-sonnet-5 rejects any non-default
                # sampling parameter (temperature/top_p/top_k) with a 400. There is no lever
                # to reduce ranking-judgment variance via sampling on this model; see git
                # history for the reverted attempt and MODEL's real behavior.
                # SYSTEM_PROMPT is static (~1550 tokens, well over the 1024-token minimum for
                # prompt caching to apply) and identical on every call — cache_control marks it
                # as reusable so repeated runs within the cache TTL (~5 min, e.g. iterative
                # testing or manual retriggers) get charged the much cheaper cache-read rate for
                # this block instead of paying full input-token price every time. The per-run
                # user_prompt (shortlist/portfolio/tracking data) is never repeated, so it isn't
                # cached — there'd be nothing to reuse.
                system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_prompt}],
            ) as stream:
                response = stream.get_final_message()
        except Exception as e:
            # The SDK already retries transient errors internally (max_retries=3 above) — this
            # catches whatever's left after those are exhausted (or a non-retryable error) and
            # degrades gracefully instead of crashing the whole pipeline run, same spirit as the
            # VIX-fetch failure handling in pipeline.py.
            return {
                "error": "Anthropic API call failed",
                "exception": str(e),
            }

        # Visibility into whether prompt caching is actually landing — cache_read_input_tokens
        # > 0 means this call reused the cached system prompt at the cheaper rate;
        # cache_creation_input_tokens > 0 means this call wrote a fresh cache entry (first call
        # in a while, or the previous one expired). Both 0 on every call would mean caching
        # isn't taking effect and is worth re-checking.
        usage = response.usage
        print(
            f"[decision_agent] token usage: input={usage.input_tokens} output={usage.output_tokens} "
            f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)} "
            f"cache_creation={getattr(usage, 'cache_creation_input_tokens', 0)}",
            file=sys.stderr,
        )

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            return json.loads(_strip_code_fence(text))
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse Claude's response as JSON",
                "truncated": response.stop_reason == "max_tokens",
                "max_tokens_used": max_tokens,
                "raw_response": text,
            }
