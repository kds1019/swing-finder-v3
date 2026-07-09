"""
Decision Agent — Anthropic API.

Pure synthesis/judgment layer on top of deterministic numbers: takes the
SmartScore shortlist + FMP research + Webull position context and produces
a final ranked, explained shortlist with flags. Never recomputes SmartScore,
sector cap, or any other numeric filter — those are already-decided facts by
the time they reach this agent.
"""

from __future__ import annotations

import json
import re
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

SYSTEM_PROMPT = """You are the final synthesis step of a swing-trading screening pipeline.
You receive tickers that have ALREADY passed deterministic scoring (SmartScore), sector-cap
filtering, and an earnings-buffer filter, each with a pre-computed trade plan (Entry/Stop/
Target/RRRatio from core/trade_plan.py — swing-low/EMA-anchored stop, Fibonacci-extension
target refined against real support/resistance). SmartScore has ALREADY been adjusted (bonus/
penalty, do not reapply) for three things: where price sits relative to its own volume profile
(PriceVsPOCPct/VolumeProfileFlag — rewarded if at/below the point of control, penalized if
stretched above it, especially "extended_above_value_area"), the ML ensemble's 5-day
directional call (MLEdgePct/MLEdgeFlag — rewarded if positive, penalized if negative or
"ml_edge_unavailable"), and the single highest-confidence detected chart pattern
(PatternName/PatternConfidence/PatternAction/PatternFlag — bullish patterns like Bull Flag,
Cup and Handle, Double Bottom, Ascending Triangle rewarded; bearish patterns like Bear Flag,
Double Top, Head and Shoulders, Descending Triangle penalized). If ml_track_record is present in
the input, it's the model's own recent directional accuracy (from actually scoring past
forecasts against what happened) — use it to calibrate how much weight the MLEdgeFlag for
THIS run deserves in your rationale (a currently-unreliable model's edge call is worth less
skepticism-adjustment than a currently-reliable one's). If ml_track_record says the sample
size is insufficient, don't speculate about accuracy — just treat MLEdgeFlag at face value.

If pick_track_record is present, it's THIS SYSTEM'S OWN historical performance — win rate
(target hit vs. stop hit) of past ranked_picks output, tracked independently of whether any
pick was actually traded. This is not about any single ticker, it's about how much to trust
the process as a whole. If pick_track_record.sufficient_data is true, weave a brief,
proportionate note into overall_recommendation reflecting it (e.g. a strong recent win rate
supports normal conviction; a weak one warrants a more conservative overall tone regardless
of how clean individual setups look this run). If sufficient_data is false, don't mention it.

Your job is ONLY to:

1. Rank the provided tickers by overall attractiveness, using the given SmartScore, trade
   plan, and research context as inputs to your judgment.
2. Explain each ranking in 1-2 sentences referencing concrete factors already provided
   (do not invent facts not present in the input, and never recompute Stop/Target/RRRatio,
   PriceVsPOCPct, MLEdgePct, or pattern fields yourself — pass them through as given).
3. Flag risks: sector concentration relative to EXISTING Webull positions (not just the
   day's shortlist), earnings-date conflicts, whether the VIX gate is open or closed, and
   ALWAYS flag if WeakRR is true (R:R fell short of the minimum after support/resistance
   refinement), StopSanityFlag is true (R:R >= 15:1 more often means an unusually tight
   stop than an unusually good target — say so explicitly, don't just repeat the number),
   VolumeProfileFlag is "extended_above_value_area" (price has run past where 70% of recent
   volume actually traded — thin support underneath), MLEdgeFlag is "negative_ml_edge"
   (a clean technical setup the model itself doesn't confirm — say so explicitly, this is
   exactly the kind of case that looks good on SmartScore alone but may not be worth trading),
   or PatternFlag indicates a bearish pattern (pattern_bear_flag, pattern_double_top,
   pattern_head_and_shoulders, pattern_descending_triangle) — name the pattern and its
   PatternAction explicitly, it's an independent technical signal from SmartScore/MLEdge.
4. If the VIX gate is closed (market_gate_open=false), your top-level recommendation must
   bias toward "monitor only, no new entries" regardless of individual SmartScores.

Do NOT recompute or second-guess the SmartScore, sector cap, earnings buffer, or trade
plan numbers — treat them as given. Respond with ONLY a JSON object matching this shape:
{
  "market_gate_open": bool,
  "overall_recommendation": str,
  "ranked_picks": [
    {"ticker": str, "rank": int, "smartscore": number, "entry": number, "stop": number,
     "target": number, "rr_ratio": number, "rationale": str, "flags": [str, ...]}
  ]
}"""


class DecisionAgent:
    def __init__(self, settings):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for DecisionAgent. Add it to your .env.")
        self.settings = settings
        self.client = Anthropic(api_key=settings.anthropic_api_key)

    def _build_user_prompt(
        self,
        smartscore_shortlist: pd.DataFrame,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        ml_track_record: Optional[dict] = None,
        pick_track_record: Optional[dict] = None,
    ) -> str:
        shortlist_records = json.loads(research_data.to_json(orient="records")) if not research_data.empty else []
        payload = {
            "market_gate_open": market_gate_open,
            "shortlist": shortlist_records,
            "existing_positions": portfolio_context.get("positions", []),
            "account_balance": portfolio_context.get("balance", {}),
            "existing_sector_exposure": portfolio_context.get("sector_exposure", {}),
            "ml_track_record": ml_track_record,
            "pick_track_record": pick_track_record,
        }
        return json.dumps(payload, default=str, indent=2)

    def synthesize(
        self,
        smartscore_shortlist: pd.DataFrame,
        research_data: pd.DataFrame,
        portfolio_context: dict,
        market_gate_open: bool,
        ml_track_record: Optional[dict] = None,
        pick_track_record: Optional[dict] = None,
    ) -> dict:
        user_prompt = self._build_user_prompt(
            smartscore_shortlist, research_data, portfolio_context, market_gate_open,
            ml_track_record, pick_track_record,
        )

        # Scaled to shortlist size. 800/ticker + 1500 overhead was enough before pattern
        # detection and the ML track record were added to the prompt — those gave the model
        # more to discuss per ticker (pattern name/confidence/action, track-record-calibrated
        # MLEdgeFlag skepticism) and a 5-ticker shortlist truncated at the resulting 5500-token
        # budget in testing. 1200/ticker + 2000 overhead leaves real headroom; capped at 24000
        # as a sanity ceiling.
        num_tickers = len(smartscore_shortlist)
        max_tokens = min(24000, max(6000, 1200 * num_tickers + 2000))

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
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
