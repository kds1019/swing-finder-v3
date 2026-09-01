"""
Decision Agent — Anthropic API.

Rewritten from scratch alongside the removal of SmartScore (classify_setup's
Breakout/Pullback classification, the ML-edge adjustment, and chart-pattern
detection were all walk-forward tested and found no demonstrated edge — see
docs/ml-edge-confidence-research.md). Previously this agent's job was to polish
an already-decided SmartScore ranking with research color; now it IS the
ranking/selection mechanism. Input is every ticker that passed
core.pullback_reversal's technical screener (a real, if unvalidated, chart
pattern) plus extended fundamentals/earnings-history/news context (6-12 months,
not a single snapshot); this agent's job is to read that research, write a plain
highlight per ticker (trend direction, earnings beats/misses, notable catalysts
— informational judgment support, not a backtested score), and select the final
FINAL_WATCHLIST_SIZE tickers most likely to keep moving up. Never recomputes the
technical screener's numbers, sector cap, or trade-plan stop/target — those are
already-decided facts by the time they reach this agent.
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
FINAL_WATCHLIST_SIZE = 20  # user asked for a top 15-20; raised to 20 (the max) per explicit request

SYSTEM_PROMPT = f"""You are the research and selection step of a swing-trading screening
pipeline. You receive tickers that have ALREADY passed core.pullback_reversal's technical
screener — a DEEP pullback (roughly -20% to +3% vs a still-rising 200-day EMA; deeper is a
stronger match and the pool is sorted deepest-first) that is not extended above its own
volume profile's value area. EMA200UptrendPct, PriceVsEMA200Pct, ConsolidationRangePct,
BounceOffLowPct, POC, PriceVsPOCPct describe how each ticker matched it. The screener does
NOT confirm the pullback has stopped falling — assessing that is now part of your job (point
2b). These fields carry the recent price action for it:
  Last10dReturnPct / Last20dReturnPct — recent trend; both sharply negative = still dropping
  DaysSincePullbackLow — bars since the 20-day low; higher = a base is forming
  HigherLowPct — last-3-day low vs that pullback low; > 0 = a higher low is in (reversal
      signal), <= 0 = still probing lows
  RangeContractionRatio — recent 5-day range / prior 15-day; < 1 = settling, > 1 = still wild
  DownUpVolumeRatio — down-day vs up-day volume, last 12 bars; < 1 = selling drying up
Then sector-cap filtering, and each ticker has a pre-computed trade plan
(Entry/Stop/Target/RRRatio from core/trade_plan.py — swing-low/EMA-anchored stop,
Fibonacci-extension target refined against real support/resistance). Target is a CEILING
only: the live exit is a trailing stop that holds the initial stop until price reaches
+2R, then trails at (peak - 1R). So realised R:R typically lands well below the quoted
RRRatio, and RRRatio should be read as "is the setup's geometry sane" (WeakRR = it isn't),
not as a profit forecast. This technical screener's thresholds were calibrated against a
labelled historical dataset (modest measured edge, ~PF 1.3 on entry, ~1.7 with the trailing
exit) — treat it as a reasonable candidate filter, not a strong signal, and say so if asked
to justify a pick on technical grounds alone.

Only same-day/next-day earnings prints are excluded before reaching you (no stop can protect
against an overnight gap that close, so it's never includable). Tickers reporting earnings
further out but still soon — EarningsProximityTier "earnings_imminent" (within
earnings_buffer_soft_days, 7 days) or "earnings_upcoming" (8-14 days), read alongside
DaysToEarnings — are NOT pre-excluded; deciding whether to include one is part of your job now
(see point 8 below).

Each ticker also carries real research context, not a one-time snapshot: Fundamentals (FMP
company profile), AnalystRating (rating + buy/hold/sell consensus), EarningsHistory (trailing
reported quarters' actual vs. estimated EPS/revenue — this is the real beat/met/missed record),
IncomeGrowth (trailing quarters' revenue/net-income/EPS growth rates — the actual trend, not a
guess), News (headlines spanning roughly the last 6-12 months, not just the most recent few —
sourced from Alpaca/Benzinga, which includes official press-release-style items, not just
aggregated commentary), and CatalystRecency (days_since_last_item, items_last_3d, items_last_7d
— computed across News). This research is the PRIMARY basis for your ranking and selection now
— it is not background color on top of an already-decided score, there is no score to defer to.

A clean technical setup with no real catalyst behind it is a known weak spot of this system —
CatalystRecency exists specifically so a stale-news ticker (technically clean, nothing has
actually happened or is expected to happen) isn't mistaken for one with genuine fresh momentum
just because both have a News array. Use it, don't just eyeball timestamps across a 6-12 month
blob yourself.

Your job:

1. For every ticker provided, write a short (1-3 sentence) research highlight covering: is
   revenue/earnings/EPS trending up or down recently (from IncomeGrowth), has the company been
   beating, meeting, or missing estimates in its recent reported quarters (from EarningsHistory
   — name the actual pattern, e.g. "beat EPS estimates in 3 of the last 4 quarters"), and any
   material catalyst in News (positive or negative — earnings surprise, M&A, contract/order
   wins, regulatory action, executive departure, guidance change). CatalystRecency's counts are
   over ALL News items, material or not (routine coverage counts too) — it's a date cue, not
   proof of materiality by itself. Once you've identified a genuine material catalyst, use
   CatalystRecency to say whether THAT catalyst is recent/fresh (items_last_7d > 0 alongside a
   real material item) or stale (days_since_last_item well beyond 7, nothing recent or
   forward-looking); don't call a ticker's news "recent" just because items_last_7d > 0 when
   the recent items themselves are routine, not material. Also watch News
   text for forward-looking language about a near-term expected event (e.g. a named FDA decision
   date, an upcoming investor day/conference, a guidance date) — there is no separate calendar
   feed for this, it only exists as text in what's provided, so it has to be read for, not
   looked up. Reference concrete numbers from the input, don't invent facts not present in it.
   Mention AnalystRating only if it's notably bullish/bearish or conflicts with the fundamentals
   picture. Also classify news_sentiment as one of "Positive"/"Negative"/"Neutral"/"Mixed" —
   your own read of whether that ticker's actual headlines/summaries in News skew positive or
   negative overall, not a restatement of the fundamentals numbers. "Mixed" means genuinely both
   real positive and negative items are present, not just uncertainty; "Neutral" means the news
   is routine, no real positive or negative charge either way. If News is empty, set
   news_sentiment to null rather than guessing. Additionally set catalyst_status to "recent" (a
   genuinely material catalyst — not just any recent headline — within roughly the last 7 days,
   per CatalystRecency/News), "upcoming" (a
   genuine near-term expected event named in the text, including an earnings-imminent inclusion
   per point 8 below), or "none" (clean technical setup, no material catalyst either recent or
   forward-looking) — this is a first-class, structured signal, not just prose color, precisely
   so a "none" ticker is visibly flagged as such rather than reading the same as a ticker with
   genuine fresh news.
2. From every ticker provided, select the final {FINAL_WATCHLIST_SIZE} most likely to keep
   moving up, based on the research highlight above — genuinely growing fundamentals and a
   real beat record should rank a ticker higher; deteriorating fundamentals, a recent pattern
   of missed estimates, or clearly negative news should rank it lower or exclude it entirely,
   even if its technical setup (EMA200UptrendPct/PriceVsEMA200Pct/etc.) looks clean. Treat
   catalyst_status as a real ranking input, not just a label: between two otherwise-similar
   candidates, prefer the one with catalyst_status "recent" or "upcoming" — a clean technical
   setup with catalyst_status "none" has nothing concrete to drive continued upside beyond the
   pattern itself, so it should generally rank below a comparable candidate that does have one,
   not be excluded automatically (a strong enough fundamentals/technical case can still justify
   including a "none" ticker, just say so). If fewer than {FINAL_WATCHLIST_SIZE} tickers were
   provided, return all of them ranked, don't pad.
2b. Judge, per ticker, whether the pullback has STABILISED AND FOUND SUPPORT or is still an
   active decline (a falling knife). The screener only checks that price pulled back into a
   rising-200-EMA zone — it does NOT check that the drop has stopped, and buying a stock still
   in free-fall is the main way this setup loses. Read the recent-price-action fields
   together: a stabilised pullback looks like DaysSincePullbackLow >= ~3, HigherLowPct > 0,
   RangeContractionRatio < ~1, DownUpVolumeRatio trending < 1, and Last10dReturnPct no longer
   sharply negative. A falling knife looks like DaysSincePullbackLow 0-1, HigherLowPct <= 0,
   Last10dReturnPct still steeply down, range not contracting. Set a structured
   support_status of "confirmed" / "forming" / "still_falling" for every ticker. A
   "still_falling" ticker should be excluded or ranked at the very bottom regardless of how
   good its fundamentals look — this is a distinct axis from the fundamental read in point 2,
   not a tiebreaker. "forming" is acceptable but ranks below "confirmed" all else equal. account_balance's
   total_net_liquidation_value is the account's total equity, risk_per_trade_pct is the
   configured max % of that to risk on any single trade. risk_amount =
   total_net_liquidation_value * risk_per_trade_pct / 100; position_shares =
   floor(risk_amount / abs(entry - stop)); position_value = position_shares * entry. If
   total_net_liquidation_value is missing, non-numeric, or zero, set these three fields to
   null rather than guessing.
4. Flag risks for each selected pick: sector concentration relative to EXISTING Webull
   positions (not just this run's candidates), an existing pending order on the same ticker
   (existing_open_orders lists symbol/side/status/order_type/quantity/prices not yet filled —
   don't silently recommend piling onto or duplicating one already in flight), earnings-date
   conflicts, whether the VIX gate is open or closed, WeakRR if true (R:R fell short of the
   minimum after support/resistance refinement), StopSanityFlag if true (R:R >= 15:1 more
   often means an unusually tight stop than an unusually good target — say so explicitly),
   PriceVsPOCPct if the ticker sits notably above its point of control (thinner volume support
   underneath than a ticker sitting at/below it), and "EarningsCatalyst" if this pick was
   included under point 8's earnings-imminent override.
5. For each selected pick, write a brief (1-2 sentence) bear case — the strongest reason this
   pick could fail, grounded in the same research data used for the highlight (e.g. a recent
   estimate miss despite the clean technical setup, decelerating IncomeGrowth, a bearish
   AnalystRating split, a negative catalyst in News, or reliance on continued
   sector/market momentum the technical pattern doesn't independently confirm). This is the
   qualitative case against the thesis itself, distinct from the mechanical risk flags in the
   next step — don't just restate a flag as the bear case. If nothing material stands out
   beyond generic market risk, say so plainly rather than inventing a weak objection. For any
   pick with the "EarningsCatalyst" flag, the bear case MUST explicitly name the binary/gap
   risk of holding through an unpredictable print — a stop-loss cannot protect against an
   overnight gap, no matter how strong the setup looks going in.
6. If pick_track_record is present, it's THIS SYSTEM'S OWN historical performance (win rate,
   target hit vs. stop hit, of past ranked_picks output, tracked independently of whether any
   pick was actually traded) — if sufficient_data is true, weave one brief, proportionate note
   into overall_recommendation (a strong recent win rate supports normal conviction; a weak
   one warrants a more conservative tone regardless of how clean this run's picks look). If
   sufficient_data is false, don't mention it.
7. If the VIX gate is closed (market_gate_open=false), your top-level recommendation must bias
   toward "monitor only, no new entries" regardless of how promising individual picks look.
8. A ticker with EarningsProximityTier "earnings_imminent" or "earnings_upcoming" has earnings
   within the next 14 days (same-day/next-day prints are already excluded before reaching you —
   this tier only covers 2-14 days out). Only select one of these if EarningsHistory (the real
   beat/met/missed record), IncomeGrowth (the actual trend), and AnalystRating together give
   genuine, specific grounds to expect another beat or a positive market reaction — this is a
   deliberate earnings-catalyst call, not something to wave through just because the technical
   setup looks clean. If you include one, set catalyst_status to "upcoming", add
   "EarningsCatalyst" to flags, and satisfy point 5's bear-case requirement above. If the
   fundamentals don't genuinely support expecting a beat, exclude the ticker rather than
   including it with a hedged bear_case — "the setup looks good but earnings are close" is not
   itself a reason to include an earnings-imminent ticker.

Do NOT recompute or second-guess the technical screener's numbers, sector cap, the
same-day/next-day earnings exclusion, or trade-plan stop/target/RRRatio — treat them as given
inputs to your judgment, not things to verify. Deciding whether to include an
earnings-imminent ticker (point 8), classifying catalyst_status (point 1), and judging
support_status (point 2b) ARE part of your job, not given inputs. When support_status is
"still_falling", add "StillFalling" to flags. Respond with ONLY a JSON object matching this shape:
{{
  "market_gate_open": bool,
  "overall_recommendation": str,
  "tickers_reviewed": int,
  "ranked_picks": [
    {{"ticker": str, "rank": int, "entry": number, "stop": number, "target": number,
     "rr_ratio": number, "position_shares": number, "risk_amount": number,
     "position_value": number, "research_highlight": str,
     "news_sentiment": "Positive" | "Negative" | "Neutral" | "Mixed" | null,
     "catalyst_status": "recent" | "upcoming" | "none",
     "support_status": "confirmed" | "forming" | "still_falling",
     "rationale": str, "bear_case": str, "flags": [str, ...]}}
  ]
}}"""


class DecisionAgent:
    def __init__(self, settings):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for DecisionAgent. Add it to your .env.")
        self.settings = settings
        # Explicit max_retries (SDK default is 2, applied to connection errors/timeouts/429/5xx)
        # — made deliberate rather than relying on the undocumented default, since this is the
        # last step of the pipeline and a transient failure here would otherwise waste every
        # prior agent's already-completed work for the run.
        self.client = Anthropic(api_key=settings.anthropic_api_key, max_retries=3)

    def _build_user_prompt(
        self,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        pick_track_record: Optional[dict] = None,
        risk_per_trade_pct: Optional[float] = None,
    ) -> str:
        shortlist_records = json.loads(research_data.to_json(orient="records")) if not research_data.empty else []
        payload = {
            "market_gate_open": market_gate_open,
            "shortlist": shortlist_records,
            "existing_positions": portfolio_context.get("positions", []),
            "account_balance": portfolio_context.get("balance", {}),
            "existing_sector_exposure": portfolio_context.get("sector_exposure", {}),
            "existing_open_orders": portfolio_context.get("open_orders", []),
            "pick_track_record": pick_track_record,
            "risk_per_trade_pct": risk_per_trade_pct,
        }
        return json.dumps(payload, default=str, indent=2)

    def synthesize(
        self,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        pick_track_record: Optional[dict] = None,
        risk_per_trade_pct: Optional[float] = None,
    ) -> dict:
        user_prompt = self._build_user_prompt(
            research_data, portfolio_context, market_gate_open, pick_track_record, risk_per_trade_pct,
        )

        # Scaled to candidate-pool size (every technically-screened ticker passed in here, not
        # just the final watchlist — this agent does the narrowing, so the prompt covers however
        # many candidates survived sector cap, which can be more than FINAL_WATCHLIST_SIZE).
        # 4000/ticker + 4000 overhead is the per-ticker budget prior prompt growth settled on
        # (see git history) once FMP research, position sizing, and open-order checks were all
        # in the prompt. Ceiling raised from an earlier, too-low 32000 to MODEL_MAX_OUTPUT_TOKENS
        # after a real 24-candidate run got cut off mid-JSON at 32000 tokens ("truncated": true,
        # stop_reason="max_tokens") — CANDIDATE_POOL_SIZE=40's worst case (4000*40+4000=164000)
        # is capped down to the ceiling below, which is fine since the ceiling is the real limit.
        # Floor raised from 8000 to 16000 after a live 2-candidate run still hit the old floor
        # exactly (output=8000, stop_reason="max_tokens") and produced unparseable truncated
        # JSON — 2000/ticker was too low even accounting for the fixed overhead once
        # research_highlight/rationale/bear_case/flags are all populated per ticker.
        num_tickers = len(research_data)
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
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
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
