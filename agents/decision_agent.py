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
screener (a pullback into a rising 200-day EMA that has stabilized and shown an early bounce,
confirmed not extended above its own volume profile's value area — EMA200UptrendPct,
PriceVsEMA200Pct, ConsolidationRangePct, BounceOffLowPct, POC, PriceVsPOCPct describe exactly
how each ticker matched it), sector-cap filtering, and an earnings-buffer filter, each with a
pre-computed trade plan (Entry/Stop/Target/RRRatio from core/trade_plan.py — swing-low/EMA-
anchored stop, Fibonacci-extension target refined against real support/resistance). This
technical screener is a real, specific chart pattern but has NOT been statistically validated
the way the system it replaced was found to have no edge — treat it as a reasonable candidate
filter, not a proven signal, and say so if asked to justify a pick on technical grounds alone.

Each ticker also carries real research context, not a one-time snapshot: Fundamentals (FMP
company profile), AnalystRating (rating + buy/hold/sell consensus), EarningsHistory (trailing
reported quarters' actual vs. estimated EPS/revenue — this is the real beat/met/missed record),
IncomeGrowth (trailing quarters' revenue/net-income/EPS growth rates — the actual trend, not a
guess), and News (headlines spanning roughly the last 6-12 months, not just the most recent
few). This research is the PRIMARY basis for your ranking and selection now — it is not
background color on top of an already-decided score, there is no score to defer to.

Your job:

1. For every ticker provided, write a short (1-3 sentence) research highlight covering: is
   revenue/earnings/EPS trending up or down recently (from IncomeGrowth), has the company been
   beating, meeting, or missing estimates in its recent reported quarters (from EarningsHistory
   — name the actual pattern, e.g. "beat EPS estimates in 3 of the last 4 quarters"), and any
   material catalyst in News (positive or negative — earnings surprise, M&A, contract/order
   wins, regulatory action, executive departure, guidance change). Reference concrete numbers
   from the input, don't invent facts not present in it. Mention AnalystRating only if it's
   notably bullish/bearish or conflicts with the fundamentals picture. Also classify
   news_sentiment as one of "Positive"/"Negative"/"Neutral"/"Mixed" — your own read of whether
   that ticker's actual headlines/summaries in News skew positive or negative overall, not a
   restatement of the fundamentals numbers. "Mixed" means genuinely both real positive and
   negative items are present, not just uncertainty; "Neutral" means the news is routine, no
   real positive or negative charge either way. If News is empty, set it to null rather than
   guessing.
2. From every ticker provided, select the final {FINAL_WATCHLIST_SIZE} most likely to keep
   moving up, based on the research highlight above — genuinely growing fundamentals and a
   real beat record should rank a ticker higher; deteriorating fundamentals, a recent pattern
   of missed estimates, or clearly negative news should rank it lower or exclude it entirely,
   even if its technical setup (EMA200UptrendPct/PriceVsEMA200Pct/etc.) looks clean. If fewer
   than {FINAL_WATCHLIST_SIZE} tickers were provided, return all of them ranked, don't pad.
3. Compute position sizing for each selected pick: account_balance's
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
   often means an unusually tight stop than an unusually good target — say so explicitly), and
   PriceVsPOCPct if the ticker sits notably above its point of control (thinner volume support
   underneath than a ticker sitting at/below it).
5. For each selected pick, write a brief (1-2 sentence) bear case — the strongest reason this
   pick could fail, grounded in the same research data used for the highlight (e.g. a recent
   estimate miss despite the clean technical setup, decelerating IncomeGrowth, a bearish
   AnalystRating split, a negative catalyst in News, or reliance on continued sector/market
   momentum the technical pattern doesn't independently confirm). This is the qualitative case
   against the thesis itself, distinct from the mechanical risk flags in the next step — don't
   just restate a flag as the bear case. If nothing material stands out beyond generic market
   risk, say so plainly rather than inventing a weak objection.
6. If pick_track_record is present, it's THIS SYSTEM'S OWN historical performance (win rate,
   target hit vs. stop hit, of past ranked_picks output, tracked independently of whether any
   pick was actually traded) — if sufficient_data is true, weave one brief, proportionate note
   into overall_recommendation (a strong recent win rate supports normal conviction; a weak
   one warrants a more conservative tone regardless of how clean this run's picks look). If
   sufficient_data is false, don't mention it.
7. If the VIX gate is closed (market_gate_open=false), your top-level recommendation must bias
   toward "monitor only, no new entries" regardless of how promising individual picks look.

Do NOT recompute or second-guess the technical screener's numbers, sector cap, earnings
buffer, or trade-plan stop/target/RRRatio — treat them as given inputs to your judgment, not
things to verify. Respond with ONLY a JSON object matching this shape:
{{
  "market_gate_open": bool,
  "overall_recommendation": str,
  "tickers_reviewed": int,
  "ranked_picks": [
    {{"ticker": str, "rank": int, "entry": number, "stop": number, "target": number,
     "rr_ratio": number, "position_shares": number, "risk_amount": number,
     "position_value": number, "research_highlight": str,
     "news_sentiment": "Positive" | "Negative" | "Neutral" | "Mixed" | null,
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
        # 2000/ticker + 3000 overhead is the per-ticker budget prior prompt growth settled on
        # (see git history) once FMP research, position sizing, and open-order checks were all
        # in the prompt. Ceiling raised from an earlier, too-low 32000 to MODEL_MAX_OUTPUT_TOKENS
        # after a real 24-candidate run got cut off mid-JSON at 32000 tokens ("truncated": true,
        # stop_reason="max_tokens") — CANDIDATE_POOL_SIZE=40's worst case (2000*40+3000=83000)
        # fits comfortably under this.
        num_tickers = len(research_data)
        max_tokens = min(MODEL_MAX_OUTPUT_TOKENS, max(8000, 2000 * num_tickers + 3000))

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
